#!/usr/bin/env python3
"""多 adapter 的正确性与资源实验：python3 lab.py [--budget-mib 64].
依赖 NumPy；分组矩阵乘法为 CPU 参考实现，内存大小是教学预算，不是实测 GPU。
检查独立逐请求结果、混合 rank、行顺序恢复、原子准入、活动权重保护与 cache key。
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import numpy as np

@dataclass(frozen=True)
class Identity:
    name: str
    revision: str
    scale: float = 1.

@dataclass
class Adapter:
    key: Identity
    a: np.ndarray
    b: np.ndarray
    @property
    def rank(self): return self.a.shape[0]

def grouped_forward(x, w, ids, adapters):
    """base 一次 GEMM；adapter 分组后写回原行号。"""
    y = x @ w.T
    for key in dict.fromkeys(ids):
        rows = np.array([i for i, k in enumerate(ids) if k == key])
        adapter = adapters[key]
        z = x[rows] @ adapter.a.T
        y[rows] += key.scale * z @ adapter.b.T
    return y

def prefix_key(tokens, identity, base_revision='base-v1', positions=None,
               rope_config='default', composition=(), visibility='causal'):
    """缓存的语义身份；实际服务还应纳入多模态输入和租户隔离范围。"""
    if positions is None: positions = list(range(len(tokens)))
    payload = dict(tokens=list(tokens), positions=list(positions), base=base_revision,
        adapter=[identity.name,identity.revision,identity.scale], rope=rope_config,
        composition=list(composition), visibility=visibility)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class CapacityError(RuntimeError): pass

class Residency:
    """MiB 教学预算：原子预留 adapter + 请求 KV；仅淘汰无引用的权重。"""
    def __init__(self, budget):
        self.budget = budget
        self.weights = {}  # immutable identity -> size MiB
        self.last_used = {}
        self.active = {}   # request -> (identity, KV MiB)
        self.clock = 0
    @property
    def used(self): return sum(self.weights.values()) + sum(v[1] for v in self.active.values())
    def admit(self, request, key, weight_mib, kv_mib):
        if request in self.active: raise ValueError('duplicate request')
        if weight_mib < 0 or kv_mib < 0: raise ValueError('negative size')
        if key in self.weights and self.weights[key] != weight_mib:
            raise ValueError('same immutable identity with a different size')
        # 先做计划；失败不会改变现有 residency。
        planned = self.weights.copy()
        required = kv_mib + (0 if key in planned else weight_mib)
        pinned = {item[0] for item in self.active.values()} | {key}
        evicted = []
        idle = sorted((k for k in planned if k not in pinned), key=lambda k:self.last_used[k])
        free = self.budget - self.used
        while free < required and idle:
            victim = idle.pop(0)
            free += planned.pop(victim)
            evicted.append(victim)
        if free < required: raise CapacityError('queue request: weights + KV do not fit')
        planned[key] = weight_mib
        self.weights = planned
        for victim in evicted: self.last_used.pop(victim)
        self.clock += 1; self.last_used[key] = self.clock
        self.active[request] = (key,kv_mib)
        self.check()
        return evicted
    def release(self, request):
        key,_ = self.active.pop(request)
        self.clock += 1; self.last_used[key] = self.clock
        self.check()
    def check(self):
        assert self.used <= self.budget
        assert all(k in self.weights for k,_ in self.active.values())
        assert set(self.weights) == set(self.last_used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-mib',type=int,default=64)
    args = parser.parse_args()
    if args.budget_mib < 1: parser.error('budget must be positive')
    rng = np.random.default_rng(2026)
    w=rng.normal(size=(6,8)); x=rng.normal(size=(5,8))
    keys=[Identity('A','v1'),Identity('B','v1',.5),Identity('C','v1',2.)]
    ranks=[2,4,1]
    adapters={k:Adapter(k,rng.normal(size=(r,8)),rng.normal(size=(6,r))) for k,r in zip(keys,ranks)}
    ids=[keys[1],keys[0],keys[2],keys[0],keys[1]]
    actual=grouped_forward(x,w,ids,adapters)
    # 独立参考：每个请求先显式合并 W，后 GEMV，不用分组路径。
    reference=np.stack([row@(w+k.scale*adapters[k].b@adapters[k].a).T for row,k in zip(x,ids)])
    error=np.max(abs(actual-reference)); assert error < 1e-12
    reversed_actual=grouped_forward(x[::-1],w,ids[::-1],adapters)
    assert np.allclose(reversed_actual,reference[::-1])
    print(f'grouped / merged-row reference / restored order: PASS (max error {error:.2e})')
    useful=sum(adapters[k].rank for k in ids); padded=len(ids)*max(ranks)
    print(f'rank lanes: useful={useful}, max-rank padding={padded}, wasted={padded-useful}')
    prefix=(2,5,7)
    same=prefix_key(prefix,keys[0]); assert same==prefix_key(prefix,keys[0])
    variants=[prefix_key(prefix,keys[1]),prefix_key(prefix,Identity('A','v2')),
        prefix_key(prefix,Identity('A','v1',2.)), prefix_key(prefix,keys[0],base_revision='base-v2'),
        prefix_key(prefix,keys[0],positions=[8,9,10]),prefix_key(prefix,keys[0],rope_config='yarn-4'),
        prefix_key(prefix,keys[0],composition=(('other','v3',.25),)),
        prefix_key(prefix,keys[0],visibility='window-4')]
    assert len(set(variants+[same]))==9
    print('prefix identity: adapter / revision / scale / base / positions / RoPE / composition / mask isolated')
    a,b,c,d=[Identity(n,'v1') for n in 'ABCD']
    pool=Residency(64)
    pool.admit('running-A',a,16,12);pool.admit('done-B',b,16,4);pool.release('done-B')
    assert pool.used==44
    evicted=pool.admit('new-C',c,32,4)
    assert evicted==[b] and pool.used==64
    before=(dict(pool.weights),dict(pool.active),dict(pool.last_used),pool.clock)
    try: pool.admit('blocked-D',d,16,4)
    except CapacityError: pass
    else: raise AssertionError('must not evict active A or C')
    assert before==(pool.weights,pool.active,pool.last_used,pool.clock)
    pool.release('running-A')
    assert pool.admit('new-D',d,16,4)==[a] and pool.used==56
    print('64 MiB: evict idle B; pin A/C; failed admission is atomic; release A then admit D: PASS')
    # 可修改预算的同一请求序列；失败请求留在队列，不计作完成。
    demo=Residency(args.budget_mib)
    admitted=blocked=0
    for i,(key,size,kv) in enumerate([(a,16,12),(b,16,4),(c,32,4),(d,16,4)]):
        try: demo.admit(f'r{i}',key,size,kv); admitted+=1
        except CapacityError: blocked+=1
    print(f'budget={args.budget_mib} MiB: admitted={admitted}, queued={blocked}, used={demo.used} MiB')
    # 时延只做单位计算，不代表硬件实测；MiB / GiB/s -> ms。
    load_ms=16/1024/16*1000
    assert load_ms==.9765625
    print(f'teaching link: 16 MiB at 16 GiB/s -> {load_ms:.6f} ms minimum payload time')
    print('PASS: mixed adapter correctness, memory accounting, lifetime and cache isolation')

if __name__=='__main__':main()
