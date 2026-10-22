#!/usr/bin/env python3
"""CPU Engram mechanisms: gates, local state, causal convolution, prefetch.
Run: python3 lab.py --seed 7 --chunk 7
Only NumPy is required. Synthetic tensors isolate implementation contracts.
"""
import argparse
import math
import numpy as np


def rms(x, eps=1e-6):
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)


def gate(q,k,eps=1e-6,demo=False):
    score=float(rms(q,eps)@rms(k,eps)/math.sqrt(q.size))
    if demo:score=np.sign(score)*math.sqrt(max(abs(score),1e-6))
    return float(1/(1+math.exp(-score)))


def hashes(tokens, n=3, rows=1009):
    # Small exact integers implement the demo's multiply-XOR structure.
    # Production fixed-width overflow conventions must be specified separately.
    multipliers=[31,131,521]
    result=[]
    for t in range(len(tokens)):
        value=0
        for j in range(n):
            value ^= (int(tokens[t-j]) if t>=j else 0)*multipliers[j]
        result.append(value%rows)
    return np.asarray(result)


def streamed_hashes(tokens,chunk,n=3):
    state=np.empty(0,dtype=np.int64);out=[]
    for start in range(0,len(tokens),chunk):
        new=tokens[start:start+chunk]
        combined=np.concatenate([state,new])
        out.extend(hashes(combined,n)[len(state):])
        state=combined[-(n-1):]
    return np.array(out)


def dense_convolution(x,weights,dilation):
    # Explicit per-channel causal matrices form an independent reference.
    T,d=x.shape;out=np.empty_like(x)
    for c in range(d):
        matrix=np.zeros((T,T))
        for j in range(len(weights)):
            offset=j*dilation
            if offset<T:
                matrix[np.arange(offset,T),np.arange(T-offset)]=weights[j,c]
        out[:,c]=matrix@x[:,c]
    return out


def streamed_convolution(x,weights,dilation,chunk):
    past=(len(weights)-1)*dilation;state=np.empty((0,x.shape[1]));out=[]
    for start in range(0,len(x),chunk):
        new=x[start:start+chunk];joined=np.concatenate([state,new]);base=len(state)
        for t in range(base,len(joined)):
            y=np.zeros(x.shape[1])
            for j,w in enumerate(weights):
                source=t-j*dilation
                if source>=0:y+=w*joined[source]
            out.append(y)
        state=joined[-past:] if past else np.empty((0,x.shape[1]))
    return np.array(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--chunk',type=int,default=7)
    args=ap.parse_args()
    if args.chunk<1:ap.error('chunk must be positive')
    rng=np.random.default_rng(args.seed)
    same=gate(np.array([1.,0.]),np.array([1.,0.]),eps=0)
    demo=gate(np.array([1.,0.]),np.array([1.,0.]),eps=0,demo=True)
    assert np.isclose(same,1/(1+math.exp(-math.sqrt(2))))
    assert gate(np.array([0.,1.]),np.array([1.,0.]),eps=0)==.5
    print(f'Aligned two-dimensional gate: paper={same:.8f}; official-demo transform={demo:.8f}')
    q=rng.normal(size=7);k=rng.normal(size=7);v=rng.normal(size=7);readout=rng.normal(size=7)
    eps=1e-6;d=q.size;rq=math.sqrt(float(np.mean(q*q))+eps);rk=math.sqrt(float(np.mean(k*k))+eps)
    a=gate(q,k,eps)
    ds=k/(rq*rk*math.sqrt(d))-(q@k)*q/(d*rq**3*rk*math.sqrt(d))
    analytical=(readout@v)*a*(1-a)*ds
    step=1e-5
    def loss(query):return float(readout@(gate(query,k,eps)*v))
    finite=np.array([(loss(q+step*np.eye(d)[i])-loss(q-step*np.eye(d)[i]))/(2*step) for i in range(d)])
    assert np.allclose(analytical,finite,atol=1e-9,rtol=1e-7)
    print('RMS-normalized gate gradient finite-difference max error:',float(np.max(np.abs(finite-analytical))))
    # A fixed canonical-ID mapping is applied before all chunking variants.
    raw=rng.integers(0,20,size=37);canonical=np.arange(20)//2;tokens=canonical[raw]
    reference_hash=hashes(tokens)
    for chunk in [1,4,args.chunk]:
        assert np.array_equal(reference_hash,streamed_hashes(tokens,chunk))
    print('Suffix hashes: whole sequence and chunked streams match exactly.')
    values=rng.normal(size=(37,5));inputs=rms(values);weights=rng.normal(size=(4,5));dilation=3
    reference=dense_convolution(inputs,weights,dilation)
    for chunk in [1,4,args.chunk]:
        streamed=streamed_convolution(inputs,weights,dilation,chunk)
        assert np.allclose(reference,streamed,atol=1e-14)
    # Boundary reset counterexample: a later chunk has nonzero older contributions.
    reset=np.concatenate([dense_convolution(inputs[:7],weights,dilation),dense_convolution(inputs[7:],weights,dilation)])
    assert np.max(np.abs(reset-reference))>1e-3
    output=values+reference/(1+np.exp(-reference))
    output_stream=values+streamed/(1+np.exp(-streamed))
    assert np.allclose(output,output_stream,atol=1e-14)
    print(f'Causal convolution: history={3*(4-1)} vectors; stream max error={np.max(np.abs(reference-streamed)):.3g}; reset counterexample detected.')
    capacity=100_000_000_000*2;payload=512*8*128*2
    assert capacity/1e9==200 and payload/2**20==1
    fetch=8+payload/25e9*1e6;before=3*15;stall=max(0,fetch-before)
    assert np.isclose(stall,4.94304)
    needed=math.ceil((8+2*payload/25e9*1e6)/15)+1
    assert needed==8
    print(f'Host table=200 GB; step payload=1 MiB; fetch={fetch:.5f} us; block4 stall={stall:.5f} us; 2 MiB hides at block{needed}.')
    print('All checks passed.')


if __name__=='__main__':
    main()
