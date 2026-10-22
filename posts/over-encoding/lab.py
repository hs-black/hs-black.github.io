#!/usr/bin/env python3
"""NumPy-only OE address, collision, sharding and byte-accounting laboratory.
Run: python3 lab.py --seed 7 --ranks 4
The vectors and access stream are synthetic; no model download or GPU needed.
"""
import argparse
import math
import numpy as np


def bounded_ngram(ids, index, n, vocab, rows):
    value = 0
    for j in range(index-n+1, index+1):
        token = int(ids[j]) if j >= 0 else 0
        value = (value*vocab+token) % rows
    return value


def full_integer_ngram(ids, index, n, vocab):
    return sum((int(ids[index-j]) if index >= j else 0)*vocab**j for j in range(n))


def sharded_lookup(table, ids, ranks):
    shards = [table[rank::ranks].copy() for rank in range(ranks)]
    result = np.empty((len(ids), table.shape[1]), dtype=table.dtype)
    for owner, shard in enumerate(shards):
        positions = np.flatnonzero(ids % ranks == owner)
        result[positions] = shard[ids[positions]//ranks]
    return result, shards


def sharded_gradient(ids, grad_out, rows, ranks):
    grads = [np.zeros((len(range(rank,rows,ranks)),grad_out.shape[1])) for rank in range(ranks)]
    for owner in range(ranks):
        positions = np.flatnonzero(ids % ranks == owner)
        np.add.at(grads[owner],ids[positions]//ranks,grad_out[positions])
    result=np.empty((rows,grad_out.shape[1]))
    for owner,shard in enumerate(grads):
        result[owner::ranks]=shard
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--ranks',type=int,default=4)
    args=ap.parse_args()
    if not 1 <= args.ranks <= 32:
        ap.error('ranks must be between 1 and 32')
    rng=np.random.default_rng(args.seed)
    tokens=rng.integers(0,131072,30)
    for n in [1,2,3,12]:
        for t in range(len(tokens)):
            assert bounded_ngram(tokens,t,n,131072,1009)==full_integer_ngram(tokens,t,n,131072)%1009
    print('Bounded modular arithmetic matches arbitrary-precision integers for orders 1,2,3,12.')
    # Fixed suffix, varying prefix: modulus equal to radix destroys all prefix information.
    rows_bad=[bounded_ngram([prev,2],1,2,5,5) for prev in range(5)]
    rows_good=[bounded_ngram([prev,2],1,2,5,7) for prev in range(5)]
    assert rows_bad==[2]*5 and len(set(rows_good))==5
    print('M=V=5:',rows_bad,'; M=7:',rows_good)
    trials,patterns,rows=2000,1000,1000
    occupancy=np.array([len(np.unique(rng.integers(rows,size=patterns))) for _ in range(trials)])
    expected=rows*(1-(1-1/rows)**patterns)
    stderr=occupancy.std(ddof=1)/math.sqrt(trials)
    assert abs(occupancy.mean()-expected)<6*stderr
    print(f'Uniform hash occupancy: analytic={expected:.3f}; mean={occupancy.mean():.3f}; SE={stderr:.3f}')
    table=rng.normal(size=(101,8));ids=rng.integers(101,size=1000)
    # Force collisions with both positive and negative gradients.
    ids[:3]=3
    grad_out=rng.normal(size=(1000,8))
    grad_out[0,:2]=[1,-2];grad_out[1,:2]=[-.5,1]
    got,_=sharded_lookup(table,ids,args.ranks)
    assert np.array_equal(got,table[ids])
    expected_grad=np.zeros_like(table)
    np.add.at(expected_grad,ids,grad_out)
    got_grad=sharded_gradient(ids,grad_out,len(table),args.ranks)
    assert np.array_equal(got_grad,expected_grad)
    # The loss is linear in selected rows; finite differences independently check accumulation.
    eps=1e-5
    def loss(e):return float(np.sum(e[ids]*grad_out))
    plus,minus=table.copy(),table.copy()
    plus[3,0]+=eps;minus[3,0]-=eps
    derivative=(loss(plus)-loss(minus))/(2*eps)
    assert np.isclose(derivative,got_grad[3,0],atol=1e-8)
    perm=rng.permutation(len(ids))
    reordered=sharded_gradient(ids[perm],grad_out[perm],len(table),args.ranks)
    assert np.allclose(reordered,expected_grad,atol=1e-13)
    print(f'{args.ranks} row shards: forward max error=0; gradient max error=0; finite difference passed.')
    capacity=2*10_000_000*128*2
    gather=7/8*capacity
    sparse=7/8*8192*2*(8+128*2)
    assert capacity==5_120_000_000 and gather==4_480_000_000
    assert sparse/2**20==3.609375
    print(f'Capacity={capacity/1e9:.2f} GB; gather={gather/1e9:.2f} GB/rank; sparse forward={sparse/2**20:.6f} MiB/rank')
    assert 2*4096*256==2_097_152
    print('All checks passed.')


if __name__=='__main__':
    main()
