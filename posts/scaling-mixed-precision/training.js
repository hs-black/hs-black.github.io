(() => {
  'use strict';
  const $=id=>document.getElementById(id), levels=[0,.5,1,1.5,2,3,4,6];
  const transpose=a=>a[0].map((_,j)=>a.map(row=>row[j]));
  function round(v){const sign=v<0?-1:1,x=Math.abs(v);let best=0;for(let i=1;i<levels.length;i++){const a=Math.abs(levels[i]-x),b=Math.abs(levels[best]-x);if(a<b-1e-12||(Math.abs(a-b)<1e-12&&i%2===0))best=i;}return sign*levels[best];}
  function quant(a,square){const y=a.map(r=>r.slice());for(let r=0;r<2;r+=square?2:1){const rows=square?[0,1]:[r],max=Math.max(...rows.flatMap(i=>a[i].map(Math.abs))),scale=max/6||1;for(const i of rows)for(let j=0;j<2;j++)y[i][j]=round(a[i][j]/scale)*scale;}return y;}
  function matrix(title,a,classes){return '<div class="mp-matrix"><h4>'+title+'</h4><div class="mp-matrix-values">'+a.flatMap((row,i)=>row.map((v,j)=>'<span class="mp-cell '+(classes?classes(i,j):'')+'">'+(v==null?'…':Number.isInteger(v)?v:v.toFixed(3))+'</span>')).join('')+'</div></div>';}
  function drawTranspose(_,step){
    const a=+$('transpose-amax').value,square=$('transpose-group').value==='square',w=[[6,1.2],[2.3,a]],left=transpose(quant(w,square)),right=quant(transpose(w),square),diff=Math.sqrt(left.flat().reduce((s,v,i)=>s+(v-right.flat()[i])**2,0));
    $('transpose-amax-value').textContent=a.toFixed(1);
    const hidden=[[null,null],[null,null]],cls=(i,j)=>Math.abs(left[i][j]-right[i][j])>1e-8?'changed':'';
    $('transpose-matrices').innerHTML=matrix('原始权重 W',w,()=>step===1?(square?'group-square':'group-row'):'')+matrix('量化 → 转置 Q(W)ᵀ',step>=2?left:hidden,step===3?cls:null)+matrix('转置 → 量化 Q(Wᵀ)',step>=3?right:hidden,step===3?cls:null);
    $('transpose-result').textContent=step===0?'先固定 W。两条路径最终都需要 W 的转置布局。':step===1?(square?'四个元素共享 scale = '+(Math.max(...w.flat().map(Math.abs))/6).toFixed(4)+'；转置保留同一个块。':'每行共享一个 scale。转置之后，每一行包含的成员发生变化。'):step===2?'左侧路径先量化 W，再把反量化后的数值转置。下一步独立构造右侧路径。':'两条路径的 Frobenius 差 = '+diff.toFixed(6)+'。'+(square?'同一组成员与确定性舍入，让这两种读向保持相同数值。':'描边格标出差异；改变右下角幅值，会改变共享 scale 和舍入刻度。');
    $('transpose-demo').dataset.difference=diff;
  }
  const tr=ArticleSteps.mount('#transpose-demo',{duration:1700,steps:[{title:'读取同一个 W',caption:'前向和反向需要相应的两种读向。'},{title:'确定 scale 的分组',caption:'分组成员决定 amax，也决定量化刻度。'},{title:'先量化再转置',caption:'将量化后的矩阵转置，保持其已有数值。'},{title:'先转置再量化',caption:'按新布局重新决定分组，逐元素比较两条路径。'}],render:drawTranspose});
  ['transpose-group','transpose-amax'].forEach(id=>$(id).addEventListener('input',()=>{tr.pause();tr.render();}));
  function drawGather(_,step){
    const amax=[6,+$('gather-amax').value],sync=$('gather-sync').checked,global=Math.max(...amax),sg=global/(448*6);$('gather-amax-value').textContent=amax[1];
    // slider uses multiples of six; these specific local scale ratios are exactly
    // represented in E4M3 only at some points, so explicitly round the block scale.
    function e4(v){let best=0,code=0,err=Infinity;for(let e=0;e<16;e++)for(let m=0;m<8;m++){if(e===15&&m===7)continue;const value=e===0?m/8*2**(-6):(1+m/8)*2**(e-7),d=Math.abs(v-value),c=e*8+m;if(d<err-1e-12||(Math.abs(d-err)<1e-12&&c%2===0)){err=d;best=value;code=c;}}return best;}
    const states=amax.map(a=>{const own=(sync?global:a)/(448*6),sb=e4(a/(6*own)),q=round(a/(sb*own));return {own,sb,q,decoded:q*sb*sg};});
    $('gather-ranks').innerHTML=states.map((x,i)=>'<section class="mp-rank"><h4>Rank '+i+'</h4><dl><dt>本地 amax</dt><dd>'+amax[i]+'</dd><dt>量化使用的 s_global</dt><dd>'+(step>=1?x.own.toFixed(6):'待同步')+'</dd><dt>FP8 块 scale</dt><dd>'+(step>=2?x.sb:'待算')+'</dd><dt>FP4 最大元素 q</dt><dd>'+(step>=2?x.q:'待算')+'</dd><dt>接收端恢复值</dt><dd class="mp-result">'+(step>=3?x.decoded.toFixed(3):'待 gather')+'</dd></dl></section>').join('');
    $('gather-result').textContent=step===0?'两个分片的最大值不同，先建立共同的张量级尺度。':step===1?(sync?'max all-reduce 得到 global amax = '+global+'，两个 rank 共同采用 s_global = '+sg.toFixed(6)+'。':'当前各自使用本地 amax，形成两套不同的张量级尺度。'):step===2?'按当前尺度生成每块的 q 和 s_block。通信同时携带 payload 与块 scale。':'接收端统一使用 s_global = '+sg.toFixed(6)+'；Rank 0 最大元素恢复为 '+states[0].decoded.toFixed(3)+'，原值为 6。'+(sync?'剩余差异由块 scale 和元素舍入决定。':'各分片的全局尺度不同，拼接后一个共同尺度会改变数值解释。');
    $('gather-demo').dataset.rank0Value=states[0].decoded;
  }
  const gather=ArticleSteps.mount('#gather-demo',{duration:1700,steps:[{title:'局部统计',caption:'每个 rank 只持有本地 token 片段。'},{title:'同步共同 scale',caption:'对 amax 做 max reduction，再计算张量级反量化乘数。'},{title:'本地 scale 与 cast',caption:'块 scale 由本地块与共同张量尺度计算。'},{title:'All-gather 与解释',caption:'接收端按 payload、块 scale 与共同 global scale 恢复数值。'}],render:drawGather});
  ['gather-amax','gather-sync'].forEach(id=>$(id).addEventListener('input',()=>{gather.pause();gather.render();}));
})();
