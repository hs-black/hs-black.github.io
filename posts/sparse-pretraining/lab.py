#!/usr/bin/env python3
"""Reference algebra for sparse training; NumPy and CPU only."""
import itertools
import numpy as np

def check(a,b,name,atol=1e-9):
    np.testing.assert_allclose(a,b,rtol=1e-9,atol=atol,err_msg=name)
    print("PASS:",name)

def top2(w):
    mask=np.zeros_like(w)
    for i,row in enumerate(w):
        mask[i,np.argsort(-np.abs(row),kind="stable")[:2]]=1
    return mask

def numerical_gradient(f,x,eps=1e-6):
    result=np.empty_like(x)
    for idx in np.ndindex(x.shape):
        plus=x.copy();minus=x.copy()
        plus[idx]+=eps;minus[idx]-=eps
        result[idx]=(f(plus)-f(minus))/(2*eps)
    return result

def transpose_masks():
    pairs=list(itertools.combinations(range(4),2))
    masks=[]
    for choice in itertools.product(pairs,repeat=4):
        m=np.zeros((4,4))
        for r,ids in enumerate(choice): m[r,list(ids)]=1
        if np.all(m.sum(axis=0)==2):masks.append(m)
    return masks

def train(mode,seed=8):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(4,64))
    target=rng.normal(size=(3,4))@x
    w=rng.normal(scale=.2,size=(3,4))
    initial=top2(w)
    previous=initial
    flips=0
    for step in range(200):
        m=initial if mode=="fixed" else top2(w)
        flips+=np.count_nonzero(m!=previous)
        residual=(m*w)@x-target
        raw=residual@x.T/x.shape[1]
        grad=m*raw if mode=="fixed" else raw
        if mode=="srste":grad=grad+.2*(1-m)*w
        w-=.03*grad
        previous=m
    m=initial if mode=="fixed" else top2(w)
    loss=np.mean(np.sum(((m*w)@x-target)**2,axis=0))/2
    assert np.isfinite(loss) and np.all(m.sum(axis=1)==2)
    return loss,flips

def main():
    rng=np.random.default_rng(4)
    w=rng.normal(size=(3,4));x=rng.normal(size=(4,5));g=rng.normal(size=(3,5))
    # Scalar loops do not reuse the matrix identities being checked.
    y=np.array([[sum(w[i,k]*x[k,n] for k in range(4)) for n in range(5)] for i in range(3)])
    dx=np.array([[sum(w[i,k]*g[i,n] for i in range(3)) for n in range(5)] for k in range(4)])
    dw=np.array([[sum(g[i,n]*x[k,n] for n in range(5)) for k in range(4)] for i in range(3)])
    check(w@x,y,"forward GEMM vs scalar sums")
    check(w.T@g,dx,"input-gradient GEMM vs scalar sums")
    check(g@x.T,dw,"weight-gradient GEMM vs scalar sums")
    m=top2(w)
    loss=lambda weights:np.sum((m*weights)@x*g)
    check(numerical_gradient(loss,w),m*(g@x.T),"fixed-mask gradient vs finite differences",1e-8)
    input_loss=lambda inputs:np.sum(((m*w)@inputs)*g)
    check(numerical_gradient(input_loss,x),(m*w).T@g,"same-mask input gradient vs finite differences",1e-8)
    raw=np.array([-2.,-4.]);w0=np.array([3.,1.]);m0=np.array([1.,0.])
    check(w0-.1*m0*raw,[3.2,1.],"fixed update")
    check(w0-.1*raw,[3.2,1.4],"STE update")
    check(w0-.1*(raw+.5*(1-m0)*w0),[3.2,1.35],"SR-STE update")

    masks=transpose_masks()
    assert len(masks)==90
    # Independent enumeration over all 16-bit masks, rather than row combinations.
    brute=[]
    for ids in itertools.combinations(range(16),8):
        bits=np.zeros(16,dtype=int);bits[list(ids)]=1;mat=bits.reshape(4,4)
        if np.all(mat.sum(axis=0)==2) and np.all(mat.sum(axis=1)==2):brute.append(tuple(bits))
    assert {tuple(m.ravel()) for m in masks}==set(brute)
    print("PASS: 90 transposable masks match independent 8-of-16 enumeration")
    w4=np.array([[9,8,2,1],[8,7,2,1],[9,6,5,4],[8,6,4,3]])
    selected=max(masks,key=lambda m:np.sum(np.abs(w4)*m))
    print("Best transposable retained magnitude:",np.sum(w4*selected))
    assert np.any(top2(w4).sum(axis=0)>2)
    print("PASS: rowwise top-2 counterexample violates transpose format")
    cycle=np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]])
    check(np.mean(cycle!=np.tile([1,1,0,0],(4,1))),.5,"mask flip rate counts exits and entries")
    check((3*2+1)/(2*2/1.6+2+.4+1),7/5.9,"complete-step speed budget")
    check(1e9*(16+2*(4.5/4)),18.25e9,"dense states plus both packed weight directions")
    for mode in ["fixed","ste","srste"]:
        a=train(mode);b=train(mode)
        check(a,b,f"{mode} deterministic trajectory")
        print(f"{mode}: final half-MSE summed over outputs={a[0]:.6f}; cumulative flipped positions={a[1]}")
    print("Done. Losses describe this toy dataset; GPU timings require a separate measurement.")

if __name__=="__main__":main()
