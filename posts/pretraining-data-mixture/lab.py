#!/usr/bin/env python3
"""CPU domain-mixture laboratory, NumPy only, no downloaded data.
Run: python3 lab.py --seed 7 --proxy-steps 40
Controlled quadratic objectives isolate scoring, gradients and proxy transfer.
"""
import argparse
import numpy as np


def mirror_step(weights, scores, rate=1.):
    logits = np.log(weights)+rate*scores
    exps = np.exp(logits-logits.max())
    return exps/exps.sum()


def features(mixtures):
    # Third simplex coordinate is determined by the first two.
    a,b = mixtures[:,0],mixtures[:,1]
    return np.column_stack([np.ones(len(a)),a,b,a*a,a*b,b*b])


def train_proxy(mixture, means, steps):
    theta = 0.
    for _ in range(steps):
        # L_i(theta)=(theta-means[i])^2/2; per-domain gradients are explicit.
        theta -= .1 * np.dot(mixture,theta-means)
    return theta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--proxy-steps',type=int,default=40)
    args = ap.parse_args()
    if args.proxy_steps < 1:
        ap.error('proxy-steps must be positive')
    rng=np.random.default_rng(args.seed)
    excess=np.array([[1.,-1.],[.2,.2]])
    scores=np.maximum(excess,0).mean(axis=1)
    wrong=np.maximum(excess.mean(axis=1),0)
    w=mirror_step(np.array([.5,.5]),scores)
    assert np.allclose(scores,[.5,.2]) and np.allclose(wrong,[0,.2])
    assert np.isclose(w[0]/w[1],np.exp(.3))
    print('DoReMi token-clipped scores:',scores,'; weights:',np.round(w,6))
    global_mean=(100*1+10*3)/110
    assert abs(global_mean-13/11)<1e-12
    print(f'Token mean={global_mean:.6f}; mean of rank means={2.:.6f}')
    # Bilevel derivative: theta' = theta - eta * G.T @ alpha.
    # The exact unconstrained alpha derivative is -eta G grad L_target(theta').
    theta=np.array([.4,-.2]);target=np.array([-.3,.8])
    G=np.array([[1.,.2],[-.4,1.1],[.8,-.3]])
    alpha=np.array([.2,.5,.3]);g_target=theta-target
    for eta in [.2,.02,.002]:
        def objective(a):
            after=theta-eta*(a@G)
            return .5*np.sum((after-target)**2)
        after=theta-eta*(alpha@G)
        exact=-eta*(G@(after-target))
        first_order=-eta*(G@g_target)
        eps=1e-5
        finite=np.array([(objective(alpha+eps*np.eye(3)[i])-objective(alpha-eps*np.eye(3)[i]))/(2*eps) for i in range(3)])
        assert np.allclose(finite,exact,atol=1e-9)
        mix=alpha@G
        predicted=-eta*np.dot(g_target,mix)+eta**2*np.dot(mix,mix)/2
        actual=objective(alpha)-.5*np.sum(g_target**2)
        assert abs(predicted-actual)<1e-12
        # Closed form: error scales exactly as eta^2 for this quadratic.
        assert np.allclose(exact-first_order,eta**2*(G@mix))
        print(f'DoGE eta={eta:g}: first-order derivative error={np.linalg.norm(exact-first_order):.6g}; finite difference passed')
    means=np.array([-1.,.3,2.])
    train_mix=rng.dirichlet(np.ones(3),size=80)
    proxy_theta=np.array([train_proxy(a,means,args.proxy_steps) for a in train_mix])
    # Independent geometric-series solution checks the training loop.
    assert np.allclose(proxy_theta,(1-.9**args.proxy_steps)*(train_mix@means))
    y=.5*(proxy_theta-.4)**2
    coef=np.linalg.lstsq(features(train_mix),y,rcond=None)[0]
    holdout=rng.dirichlet(np.ones(3),size=200)
    holdout_theta=np.array([train_proxy(a,means,args.proxy_steps) for a in holdout])
    truth=.5*(holdout_theta-.4)**2
    estimate=features(holdout)@coef
    mse=np.mean((truth-estimate)**2)
    assert mse<1e-24
    candidates=rng.dirichlet(np.ones(3),size=5000)
    prediction=features(candidates)@coef
    recommended=int(prediction.argmin())
    candidate_theta=np.array([train_proxy(a,means,args.proxy_steps) for a in candidates])
    matched=.5*(candidate_theta-.4)**2
    assert abs(matched[recommended]-matched.min())<1e-12
    # Same predictive fit, changed target objective: proxy accuracy alone cannot
    # ensure ranking transfer. This explicit shift is a controlled counterexample.
    shifted=.5*(candidate_theta-1.2)**2
    new_best=int(shifted.argmin())
    regret=shifted[recommended]-shifted[new_best]
    assert regret >= -1e-12
    if args.proxy_steps >= 5:
        assert regret>.1
    print(f'RegMix toy: 80 trained proxies; unseen-mixture MSE={mse:.3g}')
    print('Recommended mixture for target .4:',np.round(candidates[recommended],4))
    print('Best candidate for shifted target 1.2:',np.round(candidates[new_best],4))
    print(f'Target-shift loss regret={regret:.6f}')
    exposure=200*.25/2
    cap=4*2/200
    assert exposure==25 and cap==.04
    print('Exposure example: 25 passes; four-pass constrained share=4%')
    print('All checks passed.')


if __name__=='__main__':
    main()
