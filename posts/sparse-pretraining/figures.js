(() => {
'use strict';
const $=id=>document.getElementById(id);
const w=[9,8,2,1,8,7,2,1,9,6,5,4,8,6,4,3];
let mask=Array(16).fill(0);
const pairs=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
const legal=[];
for(const a of pairs)for(const b of pairs)for(const c of pairs)for(const d of pairs){
 const m=Array(16).fill(0);
 [a,b,c,d].forEach((p,r)=>p.forEach(j=>m[4*r+j]=1));
 if([0,1,2,3].every(j=>[0,1,2,3].reduce((s,r)=>s+m[4*r+j],0)===2))legal.push(m);
}
function renderMask(){
 const rows=[0,1,2,3].map(r=>mask.slice(4*r,4*r+4).reduce((a,b)=>a+b,0));
 const cols=[0,1,2,3].map(j=>[0,1,2,3].reduce((s,r)=>s+mask[4*r+j],0));
 $('transpose-cells').innerHTML=w.map((v,i)=>'<div class="s-cell '+(mask[i]?'on':'')+'"><button type="button" data-cell="'+i+'" aria-label="第 '+(Math.floor(i/4)+1)+' 行第 '+(i%4+1)+' 列，幅值 '+v+'" aria-pressed="'+!!mask[i]+'">'+(mask[i]?'保留':'移除')+'<small>|w| = '+v+'</small></button></div>').join('');
 $('transpose-result').textContent='每行数量 ['+rows.join(', ')+']；每列数量 ['+cols.join(', ')+']。前向 '+(rows.every(v=>v<=2)?'合法':'超出 2:4')+'；转置 '+(cols.every(v=>v<=2)?'合法':'超出 2:4')+'。保留 '+mask.reduce((a,b)=>a+b,0)+' / 16 个，保留幅值总和 '+w.reduce((s,v,i)=>s+v*mask[i],0)+'。';
 $('transpose-result').dataset.mask=mask.join('');
}
$('transpose-cells').addEventListener('click',e=>{const b=e.target.closest('button[data-cell]');if(b){mask[Number(b.dataset.cell)]^=1;renderMask();$('transpose-cells').querySelector('[data-cell="'+b.dataset.cell+'"]').focus();}});
$('mask-row').addEventListener('click',()=>{mask=Array(16).fill(0);for(let r=0;r<4;r++)[0,1,2,3].sort((a,b)=>w[r*4+b]-w[r*4+a]||a-b).slice(0,2).forEach(j=>mask[4*r+j]=1);renderMask();});
$('mask-both').addEventListener('click',()=>{let best=-Infinity;for(const m of legal){const score=w.reduce((s,v,i)=>s+v*m[i],0);if(score>best){best=score;mask=m.slice();}}renderMask();});
$('mask-reset').addEventListener('click',()=>{mask=Array(16).fill(0);renderMask();});
const number=(id,min,max)=>{const raw=Number($(id).value);return $(id).value!==''&&Number.isFinite(raw)&&raw>=min&&raw<=max?raw:NaN;};
function renderTime(){
 const t=number('time-gemm',.1,10),s=number('time-speed',1,2),p=number('time-pack',0,10),o=number('time-other',0,10);
 if(![t,s,p,o].every(Number.isFinite)){
  $('dense-label').textContent='Dense：待输入';$('sparse-label').textContent='稀疏：待输入';
  $('dense-bar').style.width='0%';$('sparse-bar').style.width='0%';
  $('time-result').textContent='请输入范围内的毫秒数：每个 GEMM 为 0.1–10，打包与其他工作为 0–10。';
  $('time-result').dataset.speedup='';return;
 }
 const dense=3*t+o,sparse=2*t/s+t+p+o,max=Math.max(dense,sparse),budget=2*t-2*t/s;
 $('time-speed-value').textContent=s.toFixed(1)+'×';
 $('dense-label').textContent='Dense：'+dense.toFixed(2)+' ms';
 $('sparse-label').textContent='稀疏：'+sparse.toFixed(2)+' ms';
 $('dense-bar').style.width=dense/max*100+'%';$('sparse-bar').style.width=sparse/max*100+'%';
 $('time-result').textContent='稀疏三项 GEMM：'+(t/s).toFixed(2)+' + '+(t/s).toFixed(2)+' + '+t.toFixed(2)+' ms。总加速比 '+(dense/sparse).toFixed(3)+'×；打包收支平衡点 '+budget.toFixed(2)+' ms。当前'+(sparse<dense?'节省 '+(dense-sparse).toFixed(2):sparse>dense?'增加 '+(sparse-dense).toFixed(2):'变化 0.00')+' ms。';
 $('time-result').dataset.speedup=(dense/sparse).toFixed(10);
}
['time-gemm','time-speed','time-pack','time-other'].forEach(id=>$(id).addEventListener('input',renderTime));
$('time-reset').addEventListener('click',()=>{$('time-gemm').value=1;$('time-speed').value=2;$('time-pack').value=.2;$('time-other').value=.5;renderTime();});
$('mask-row').click();renderTime();
})();
