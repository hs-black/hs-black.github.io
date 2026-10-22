/* Numerical model-merging figures. Sliders and steps are reversible; no autoplay. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const fmt = (n, d = 2) => (Math.abs(n) < 10 ** (-d) / 2 ? 0 : n).toFixed(d);
  const vec = row => '[' + row.map(x => fmt(x)).join(', ') + ']';
  const accent = 'var(--accent)', gray = 'var(--text-faint)', orange = 'var(--accent-2)';

  function axes(svg, xmin, xmax, ymax, xticks) {
    const w = Math.max(210, svg.clientWidth), h = 245;
    const left = 37, right = 14, top = 18, bottom = 34;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const tx = x => left + (w - left - right) * (x - xmin) / (xmax - xmin);
    const ty = y => h - bottom - (h - top - bottom) * y / ymax;
    let html = '';
    for (let j = 0; j <= 4; j++) {
      const value = ymax * j / 4, y = ty(value);
      html += `<line x1="${left}" x2="${w-right}" y1="${y}" y2="${y}" stroke="var(--border)"/><text x="${left-6}" y="${y+4}" text-anchor="end" fill="var(--text-soft)" font-size="11">${fmt(value, ymax <= 1 ? 2 : 1)}</text>`;
    }
    for (const x of xticks) html += `<text x="${tx(x)}" y="${h-13}" text-anchor="middle" fill="var(--text-soft)" font-size="11">${fmt(x, xmax <= 1 ? 2 : 1)}</text>`;
    return { w, h, tx, ty, html };
  }
  function permutation() {
    const t = +el('perm-t').value, x = +el('perm-x').value, aligned = el('perm-align').checked;
    const weight = aligned ? 1 : 1 - 2*t, factor = Math.abs(weight);
    el('perm-t-out').textContent = fmt(t);
    el('perm-x-out').textContent = fmt(x, 1);
    el('perm-weights').textContent = vec([weight, -weight]);
    el('perm-output').textContent = `输入 ${fmt(x,1)} → 输出 ${fmt(factor*Math.abs(x))}，目标 ${fmt(Math.abs(x))}`;
    el('perm-loss').textContent = fmt((1-factor)**2/3, 5);
    const svg = el('perm-chart'), a = axes(svg, -2, 2, 2.2, [-2,-1,0,1,2]);
    let html = a.html;
    for (const [scale, color, dashed] of [[1,gray,true],[factor,accent,false]]) {
      html += `<path d="M${a.tx(-2)},${a.ty(2*scale)} L${a.tx(0)},${a.ty(0)} L${a.tx(2)},${a.ty(2*scale)}" fill="none" stroke="${color}" stroke-width="${dashed?5:3}" ${dashed?'stroke-dasharray="5 4"':''}/>`;
    }
    html += `<line x1="${a.tx(x)}" x2="${a.tx(x)}" y1="${a.ty(0)}" y2="${a.ty(Math.abs(x))}" stroke="${accent}" stroke-dasharray="3 4" opacity=".5"/><circle cx="${a.tx(x)}" cy="${a.ty(factor*Math.abs(x))}" r="5" fill="${accent}"/>`;
    svg.innerHTML = html;
  }

  const deltas = [[.8,-.6,.05,.02],[.7,.9,-.04,.01],[-2,.4,.03,-.06]];
  const base = [3,2,1,-1];
  function tiesData() {
    const keep = +el('ties-keep').value;
    const trimmed = deltas.map(row => {
      const indices = row.map((v,j)=>j).sort((a,b)=>Math.abs(row[b])-Math.abs(row[a])).slice(0,keep);
      return row.map((v,j)=>indices.includes(j)?v:0);
    });
    const totals = base.map((_,j)=>trimmed.reduce((sum,row)=>sum+row[j],0));
    const signs = totals.map(Math.sign);
    const selected = trimmed.map(row=>row.map((v,j)=>v!==0 && Math.sign(v)===signs[j]));
    const merged = base.map((_,j)=> {
      const values=trimmed.map((row,i)=>selected[i][j]?row[j]:null).filter(v=>v!==null);
      return values.length ? values.reduce((sum,v)=>sum+v,0)/values.length : 0;
    });
    return { trimmed, totals, signs, selected, merged };
  }
  const tiesSteps = [
    { title:'原始任务更新', caption:'每一行是一份微调模型减去共同基座后的变化；四列对应相同的四个参数。' },
    { title:'Trim：保留绝对值大的坐标', caption:'每行独立排序，保留指定数量的更新。被删除的位置变为零。' },
    { title:'Elect：按各列总和选择方向', caption:'先计算带符号的列总和，再取正负号。第一列的 −2 可以超过两个正更新的总幅度。' },
    { title:'Disjoint merge：平均同向非零更新', caption:'高亮格的方向与该列选定符号一致；每列分别除以接纳的非零更新数。' },
    { title:'加回共同基座', caption:'同向平均后的更新乘以 λ，再加到基座 θ₀。调整 λ 可以比较最后的更新幅度。' }
  ];
  let tiesStage = 0;
  function renderTies(_, index) {
    tiesStage = index;
    const {trimmed,totals,signs,selected,merged} = tiesData();
    const scale = +el('ties-scale').value;
    el('ties-scale-out').textContent = fmt(scale,1);
    const rows = index === 0 ? deltas : trimmed;
    let html = '<span class="mm-cell heading">任务</span>' + base.map((_,j)=>`<span class="mm-cell heading">参数 ${j+1}</span>`).join('');
    rows.forEach((row,i)=> {
      html += `<span class="mm-cell heading">${'ABC'[i]}</span>`;
      row.forEach((v,j)=> {
        const active = index >= 3 ? selected[i][j] : index >= 1 && v!==0;
        const dim = index>=1 && (v===0 || (index>=3 && !selected[i][j]));
        html += `<span class="mm-cell${active?' selected':''}${dim?' muted':''}">${fmt(v)}</span>`;
      });
    });
    if (index >= 2) {
      html += '<span class="mm-cell heading">总和</span>' + totals.map(v=>`<span class="mm-cell">${fmt(v)}</span>`).join('');
      html += '<span class="mm-cell heading">方向</span>' + signs.map(s=>`<span class="mm-cell">${s>0?'+':s<0?'−':'0'}</span>`).join('');
    }
    el('ties-matrix').innerHTML = html;
    const result = index === 0 ? `每行原始向量都包含 4 个坐标。接下来保留每行最大的 ${el('ties-keep').value} 个绝对值。`
      : index === 1 ? `各行保留 ${el('ties-keep').value} / 4 个坐标。下一步沿同一列相加，得到该坐标的方向。`
      : index === 2 ? `列总和：<strong>${vec(totals)}</strong>。总和为零的坐标没有选定方向。`
      : index === 3 ? `同向非零平均：<strong>${vec(merged)}</strong>。每列分别计数，零值不进入分母。`
      : `更新：${fmt(scale,1)} × ${vec(merged)}<br>最终参数：<strong>${vec(base.map((v,j)=>v+scale*merged[j]))}</strong>`;
    el('ties-result').innerHTML = result;
  }
  const stepper = ArticleSteps.mount('#ties-figure', {steps:tiesSteps, duration:1800, render:renderTies});

  let seed = 23;
  function uniforms(n) {
    let state = seed;
    return Array.from({length:n},()=> { state=(Math.imul(state,1664525)+1013904223)>>>0; return state/4294967296; });
  }
  function binomial(n,q) {
    if(q===1) return Array.from({length:n+1},(_,k)=>k===n?1:0);
    const values = [Math.pow(1-q,n)];
    for(let k=1;k<=n;k++) values.push(values[k-1]*(n-k+1)/k*q/(1-q));
    return values;
  }
  function dare() {
    const q=+el('dare-q').value, n=+el('dare-width').value;
    const count=uniforms(n).filter(v=>v<q).length;
    const stdev=Math.sqrt(.04*(1-q)/(q*n)), value=.2*count/(q*n), probs=binomial(n,q);
    el('dare-q-out').textContent=`${fmt(q*100,0)}% 保留 · ${fmt((1-q)*100,0)}% 丢弃`;
    el('dare-moments').textContent=`0.2000 / ${fmt(stdev,4)}`;
    const zero=probs[0];
    el('dare-zero').textContent=`全部丢弃概率：${zero<.00001 && zero>0 ? zero.toExponential(2) : fmt(100*zero,4)+'%'}`;
    el('dare-sample').textContent=fmt(value,4);
    el('dare-count').textContent=`保留 ${count} / ${n} 个通道；每个保留通道贡献 ${fmt(.2/(q*n),4)}`;
    // Full support is shown: the horizontal endpoint is all n channels retained.
    const xmax=.2/q, ymax=Math.min(1,Math.max(...probs)*1.15);
    const svg=el('dare-chart'), a=axes(svg,0,xmax,ymax,[0,xmax/2,xmax]);
    let html=a.html;
    const barWidth=Math.max(1,Math.min(15,(a.tx(xmax)-a.tx(0))/(n+1)*.68));
    probs.forEach((prob,k)=> {
      if(prob < 1e-12) return;
      const x=a.tx(.2*k/(q*n));
      const left=k===0?x:k===n?x-barWidth:x-barWidth/2;
      html+=`<rect x="${left}" y="${a.ty(prob)}" width="${barWidth}" height="${a.ty(0)-a.ty(prob)}" fill="${k===count?orange:accent}" opacity="${k===count?1:.68}"/>`;
    });
    html+=`<line x1="${a.tx(.2)}" x2="${a.tx(.2)}" y1="${a.ty(0)}" y2="${a.ty(ymax)}" stroke="${gray}" stroke-dasharray="4 4"/><text x="${Math.min(a.w-48,a.tx(.2)+5)}" y="13" fill="var(--text-soft)" font-size="11">均值 0.2</text>`;
    svg.innerHTML=html;
  }
  ['perm-t','perm-x','perm-align'].forEach(id=>el(id).addEventListener('input',permutation));
  ['ties-keep','ties-scale'].forEach(id=>el(id).addEventListener('input',()=>stepper.render()));
  ['dare-q','dare-width'].forEach(id=>el(id).addEventListener('input',dare));
  el('dare-resample').addEventListener('click',()=>{ seed=(seed+104729)>>>0; dare(); });
  function renderAll(){permutation();renderTies(tiesSteps[tiesStage],tiesStage);dare();}
  renderAll();
  let frame;
  window.addEventListener('resize',()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(renderAll);});
})();
