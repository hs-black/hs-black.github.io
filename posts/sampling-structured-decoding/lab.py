#!/usr/bin/env python3
"""Sampling + a tiny JSON pushdown matcher, Python 3 + NumPy.

python3 lab.py --samples 100000 --seed 7
The grammar is deliberately small: value := 0 | 1 | '[' [value (',' value)*] ']'.
It supports arbitrary nesting, multi-character tokens and token rollback;
it is not a full JSON Schema implementation. Independent json.loads checks
all generated completions. Timings here are not serving benchmarks.
"""
import argparse
import copy
import json
import math
import numpy as np

VOCAB = ['[', ']', ',', '0', '1', '[0', ',1]', '0]', '10', '<eos>']
EOS = len(VOCAB)-1


def distribution(logits, allowed=None, temperature=1., top_k=0, top_p=1.):
    """Policy: grammar mask -> temperature -> top-k -> top-p -> renormalize.

    Stable ties keep lower token id first. temperature=0 is a separate greedy
    branch. This explicit policy need not match every serving API's order.
    """
    x = np.asarray(logits,dtype=np.float64).copy()
    if np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError('NaN/+inf logits are not a probability distribution')
    if not np.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError('invalid sampling parameters')
    if allowed is not None:
        x[~np.asarray(allowed,dtype=bool)] = -np.inf
    valid = np.isfinite(x)
    if not valid.any():
        raise ValueError('no legal token: inspect matcher, stop policy and tokenizer')
    order = np.argsort(-x,kind='stable')
    out = np.zeros_like(x)
    if temperature == 0:
        out[order[0]] = 1.
        return out
    if top_k:
        valid[order[min(top_k,int(valid.sum())):]] = False
    values = x[valid]
    # At low temperature, subtract the legal maximum before dividing:
    # 1e308 / 0.1 would overflow even though its final softmax is well-defined.
    # At T >= 1, division is safe first and also avoids overflowing the
    # difference between opposite-sign extreme logits at a large temperature.
    # A negative overflow / exp underflow means negligible probability, zero.
    with np.errstate(over='ignore',under='ignore'):
        if temperature < 1:
            scaled = (values-values.max())/temperature
        else:
            scaled = values/temperature
            scaled -= scaled.max()
        p = np.zeros_like(x)
        p[valid] = np.exp(scaled)
    p /= p.sum()
    order = np.argsort(-p,kind='stable')
    count = min(len(p),int(np.searchsorted(np.cumsum(p[order]),top_p,side='left'))+1)
    out[order[:count]] = p[order[:count]]
    out /= out.sum()
    return out


class ArrayMatcher:
    """One matcher per request; a stack tracks array nesting.

    Each accepted token stores a snapshot. Mask generation tests a clone, so
    rejecting a multi-character token never corrupts the live state.
    """
    def __init__(self):
        self.root = 'value'
        self.stack = []
        self.terminated = False
        self.text = ''
        self.history = []

    def snapshot(self):
        return self.root,self.stack.copy(),self.terminated,self.text

    def restore(self,state):
        self.root,self.stack,self.terminated,self.text = copy.deepcopy(state)

    def _complete(self):
        if self.stack:
            self.stack[-1] = 'comma_or_end'
        else:
            self.root = 'done'

    def _char(self,ch):
        state = self.stack[-1] if self.stack else self.root
        if state in ('value','value_or_end'):
            if ch in '01':
                self._complete()
            elif ch == '[':
                self.stack.append('value_or_end')
            elif ch == ']' and state == 'value_or_end' and self.stack:
                self.stack.pop()
                self._complete()
            else:
                return False
        elif state == 'comma_or_end':
            if ch == ',':
                self.stack[-1] = 'value'
            elif ch == ']':
                self.stack.pop()
                self._complete()
            else:
                return False
        else:
            return False
        return True

    def accept(self,token):
        if self.terminated or not 0 <= token < len(VOCAB):
            return False
        old = self.snapshot()
        if token == EOS:
            ok = self.root == 'done' and not self.stack
            if ok:
                self.terminated = True
        else:
            ok = all(self._char(ch) for ch in VOCAB[token])
            if ok:
                self.text += VOCAB[token]
        if not ok:
            self.restore(old)
            return False
        self.history.append(old)
        return True

    def allowed(self):
        mask = []
        for token in range(len(VOCAB)):
            trial = copy.deepcopy(self)
            mask.append(trial.accept(token))
        return np.array(mask,dtype=bool)

    def rollback(self,n):
        if n < 0 or n > len(self.history):
            raise ValueError('rollback exceeds token history')
        for _ in range(n):
            self.restore(self.history.pop())


def valid_value(value):
    return type(value) is int and value in (0,1) or isinstance(value,list) and all(valid_value(v) for v in value)


def run(args):
    p = np.array([.5,.3,.2])
    # Independent temperature reference: p_i^(1/T), normalized.
    actual = distribution(np.log(p),temperature=.5)
    expected = p**2 / np.sum(p**2)
    assert np.allclose(actual,expected,atol=1e-14,rtol=0)
    nucleus = distribution(np.log(p),top_p=.75)
    assert np.allclose(nucleus,[.625,.375,0],atol=1e-14,rtol=0)
    masked = distribution(np.log([.7,.2,.1]),[False,True,True],top_k=1)
    assert np.array_equal(masked,[0,1,0])
    assert np.array_equal(distribution([1e308,0],temperature=.1),[1,0])
    assert np.array_equal(distribution([-1e308,-1e308],temperature=.1),[.5,.5])
    assert np.array_equal(distribution([0,1e308],[True,False],temperature=.1),[1,0])
    # Independent two-class logistic reference: the scaled logit gap is 2.
    extreme = distribution([1e308,-1e308],temperature=1e308)
    first = 1/(1+math.exp(-2))
    assert np.allclose(extreme,[first,1-first],atol=1e-14,rtol=0)
    for bad_temperature in (float('nan'),float('inf'),float('-inf')):
        try:
            distribution([1,2],temperature=bad_temperature)
            raise AssertionError('non-finite temperature was silently accepted')
        except ValueError:
            pass
    try:
        distribution([1,2],[False,False])
        raise AssertionError('all-masked input was silently accepted')
    except ValueError:
        pass
    rng = np.random.default_rng(args.seed)
    empirical = np.bincount(rng.choice(3,size=args.samples,p=nucleus),minlength=3)/args.samples
    # Fixed seed, and a conservative binomial tolerance (not exact equality).
    tolerance = max(.01,6*math.sqrt(.25/args.samples))
    assert np.max(np.abs(empirical-nucleus)) < tolerance
    matcher = ArrayMatcher()
    assert matcher.accept(VOCAB.index('[0'))
    saved = matcher.snapshot()
    assert not matcher.accept(VOCAB.index('10')) and matcher.snapshot() == saved
    assert matcher.accept(VOCAB.index(',1]'))
    assert matcher.text == '[0,1]'
    matcher.rollback(1)
    assert matcher.snapshot() == saved and matcher.text == '[0'
    assert matcher.accept(VOCAB.index(']')) and matcher.accept(EOS)
    assert matcher.terminated and not matcher.allowed().any()
    nested = ArrayMatcher()
    for piece in ['[', '[0', ',1]', ']']:
        assert nested.accept(VOCAB.index(piece))
    assert nested.text == '[[0,1]]' and nested.accept(EOS)
    assert json.loads(nested.text) == [[0,1]]
    trailing_comma = ArrayMatcher()
    for piece in ['[0', ',']:
        assert trailing_comma.accept(VOCAB.index(piece))
    assert not trailing_comma.accept(VOCAB.index(']'))
    completed = 0
    for _ in range(100):
        m = ArrayMatcher()
        for step in range(256):
            # After 24 steps, discourage new nesting but preserve legality.
            # A finite limit is still checked; validity alone does not force EOS.
            logits = rng.normal(size=len(VOCAB))
            if step >= 24:
                logits[[0,5]] = -100
                logits[[1,3,9]] += 8
            probs = distribution(logits,m.allowed(),top_p=.95)
            token = int(rng.choice(len(VOCAB),p=probs))
            assert m.accept(token)
            if m.terminated:
                value = json.loads(m.text)
                assert valid_value(value)
                completed += 1
                break
        else:
            raise AssertionError('generation reached token budget before EOS')
    print('PASS: temperature, top-p crossing token, grammar-before-top-k, all-masked error')
    print('top_p_reference=[0.625, 0.375, 0.000]')
    print('sample_frequencies=' + np.array2string(empirical,precision=5))
    print('PASS: token atomicity, nested stack, rollback, EOS lifecycle')
    print(f'PASS: {completed}/100 generated JSON values parsed independently')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=7)
    args = parser.parse_args()
    if args.samples < 1000:
        parser.error('--samples must be >= 1000')
    run(args)
