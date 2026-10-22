#!/usr/bin/env python3
"""Activation sparsity: exact small references with NumPy, on CPU."""
import itertools
import numpy as np

def check(a,b,name,atol=1e-10):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=atol,err_msg=name)
    print("PASS:",name)

def col24(a):
    """Top two magnitudes per group of four rows in each column."""
    result=a.copy()
    for start in range(0,a.shape[0],4):
        for j in range(a.shape[1]):
            ids=np.argsort(-np.abs(a[start:start+4,j]),kind="stable")[:2]
            for k in range(4):
                if k not in ids:result[start+k,j]=0
    return result

def main():
    h=np.array([1,.6,-.15,.08,-.03,.02,-.01,.4])
    w=np.array([[1,1,4,2,1,-1,2,-.5],[.5,-1,1,-2,3,1,-1,2]])
    opnorm=np.linalg.svd(w,compute_uv=False)[0]
    for tau in [step/20 for step in range(21)]:
        kept=np.abs(h)>=tau;sparse=h*kept
        ref=np.array([sum(w[i,j]*h[j] for j in range(8) if kept[j]) for i in range(2)])
        check(w@sparse,ref,f"threshold {tau:.2f}: matrix vs selected-column sum")
        error=np.linalg.norm(w@h-w@sparse)
        bound=opnorm*np.linalg.norm(h-sparse)
        assert error<=bound+1e-12
    print("PASS: spectral-norm error bound at every threshold")
    delta=h-h*(np.abs(h)>=.1)
    check(w@delta,[.09,-.22],"worked threshold example")
    check(np.linalg.norm(w@delta),np.sqrt(.09**2+.22**2),"worked output-error length")
    check(np.dot([1,100],[.1,.01])-np.dot([1,100],[.1,0]),1.,
          "small activation with large weight column")

    rho=.1
    # Independent exact enumeration of all 2^B events, with their probabilities.
    for batch in [1,2,4,8]:
        union_probability=0.
        for events in itertools.product([0,1],repeat=batch):
            probability=np.prod([rho if event else 1-rho for event in events])
            if any(events):union_probability+=probability
        check(union_probability,1-(1-rho)**batch,f"batch {batch}: enumerated union probability")
    rng=np.random.default_rng(12)
    independent=rng.random((32,100000))<rho
    measured=independent.any(axis=0).mean()
    expected=1-(1-rho)**32
    assert abs(measured-expected)<.003
    print(f"Sampled union at B=32: {measured:.6f}; expectation {expected:.6f}")
    shared=np.broadcast_to(independent[0],independent.shape)
    check(shared.any(axis=0).mean(),shared[0].mean(),"identical masks preserve union density")

    a=np.array([[1.,2,0,0]]*4+[[0.,0,3,4]]*4)
    p=np.array([0,4,1,5,2,6,3,7])
    check(np.sum((a-col24(a))**2),60.,"original token order loses energy 60")
    check(np.sum((a[p]-col24(a[p]))**2),0.,"interleaved order satisfies both 2:4 axes")
    # Verify the actual product after compression using an explicit sparse loop.
    weight=rng.normal(size=(4,3));compressed=col24(a)
    ref=np.zeros((8,3))
    for i,j in zip(*np.nonzero(compressed)):ref[i]+=compressed[i,j]*weight[j]
    check(compressed@weight,ref,"2:4 product vs explicit nonzero accumulation")
    x=rng.normal(size=(8,4));up=rng.normal(size=(4,7));down=rng.normal(size=(7,4))
    f=lambda v:np.maximum(v@up,0)**2@down
    check(f(x[p]),f(x)[p],"pointwise FFN commutes with token permutation")
    check(f(x[p])[np.argsort(p)],f(x),"inverse permutation restores token order")
    # Apply the backward pattern to the forward-effective activation.
    forward=a.copy();forward[0,0]=0
    backward=col24(forward)
    assert np.all(backward[forward==0]==0)
    print("PASS: backward selection does not restore forward-dropped values")
    check(120/(62+18+50-30),1.2,"prediction overlap critical-path budget")
    print("All checks passed. Sparsity counts and CPU algebra do not estimate GPU speed.")

if __name__=="__main__":main()
