(() => {
'use strict';
const $=id=>document.getElementById(id);
const h=[1,.6,-.15,.08,-.03,.02,-.01,.4];
const w=[[1,1,4,2,1,-1,2,-.5],[.5,-1,1,-2,3,1,-1,2]];
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const norm=a=>Math.sqrt(dot(a,a));
const a=dot(w[0],w[0]),b=dot(w[1],w[1]),c=dot(w[0],w[1]);
const spectral=Math.sqrt((a+b+Math.sqrt((a-b)**2+4*c*c))/2);
function renderThreshold(){
 const tau=Number($('act-threshold').value),kept=h.map(v=>Math.abs(v)>=tau),sparse=h.map((v,i)=>kept[i]?v:0);
 const denseY=w.map(row=>dot(row,h)),sparseY=w.map(row=>dot(row,sparse));
 const error=norm(denseY.map((v,i)=>v-sparseY[i])),bound=spectral*norm(h.map((v,i)=>v-sparse[i]));
 $('act-threshold-value').textContent=tau.toFixed(2);
 $('act-cells').innerHTML=h.map((v,i)=>'<div class="s-cell '+(kept[i]?'on':'dim')+'">h'+(i+1)+' = '+v.toFixed(2)+'<small>'+(kept[i]?'保留':'移除')+'</small></div>').join('');
 $('act-result').textContent='保留 '+kept.filter(Boolean).length+' / 8。完整输出 ['+denseY.map(v=>v.toFixed(3)).join(', ')+']；截断后 ['+sparseY.map(v=>v.toFixed(3)).join(', ')+']。误差长度 '+error.toFixed(4)+'；谱范数上界 '+bound.toFixed(4)+'。';
 $('act-result').dataset.error=error.toFixed(10);$('act-result').dataset.bound=bound.toFixed(10);
}
$('act-threshold').addEventListener('input',renderThreshold);
$('act-reset').addEventListener('click',()=>{$('act-threshold').value=.1;renderThreshold();});
function random(t,j){
 let x=((t+1)*73856093)^((j+1)*19349663)^0x5f3759df;
 x=Math.imul(x^(x>>>16),0x45d9f3b);x=Math.imul(x^(x>>>16),0x45d9f3b);x=(x^(x>>>16))>>>0;
 return x/4294967296;
}
function renderUnion(){
 const rho=Number($('union-density').value),batch=Number($('union-batch').value),shared=$('union-mode').value==='shared';
 $('union-token').max=batch;$('union-token').value=Math.min(Number($('union-token').value),batch);
 const token=Number($('union-token').value)-1;
 const masks=Array.from({length:batch},(_,t)=>Array.from({length:16},(_,j)=>random(shared?0:t,j)<rho));
 const counts=Array.from({length:16},(_,j)=>masks.reduce((s,row)=>s+Number(row[j]),0));
 const union=counts.filter(v=>v>0).length,total=counts.reduce((s,v)=>s+v,0),expect=shared?rho:1-(1-rho)**batch;
 $('union-density-value').textContent=(rho*100).toFixed(0)+'%';
 $('union-token-value').textContent=token+1;
 $('union-cells').innerHTML=counts.map((count,j)=>'<div class="s-cell '+(masks[token][j]?'on':count===0?'dim':'')+'">神经元 '+(j+1)+'<small>使用 '+count+'/'+batch+'</small></div>').join('');
 $('union-result').textContent='这组样本：每 token 平均活跃 '+(total/batch).toFixed(2)+' / 16；batch 共使用 '+union+' / 16（'+(union/16*100).toFixed(2)+'%）。当前关系下，并集的理论期望为 '+(expect*100).toFixed(2)+'%。';
 $('union-result').dataset.union=union;$('union-result').dataset.expectation=expect.toFixed(10);$('union-result').dataset.total=total;
}
['union-density','union-token'].forEach(id=>$(id).addEventListener('input',renderUnion));
['union-batch','union-mode'].forEach(id=>$(id).addEventListener('change',renderUnion));
$('union-reset').addEventListener('click',()=>{$('union-density').value=.1;$('union-batch').value=8;$('union-mode').value='independent';$('union-token').value=1;renderUnion();});
function renderPerm(){
 const order=$('perm-order').value==='original'?[0,1,2,3,4,5,6,7]:[0,4,1,5,2,6,3,7];
 const matrix=order.map(t=>t<4?[1,2,0,0]:[0,0,3,4]),mask=Array.from({length:8},()=>[0,0,0,0]);
 for(let start=0;start<8;start+=4)for(let j=0;j<4;j++){
  [0,1,2,3].sort((r,s)=>Math.abs(matrix[start+s][j])-Math.abs(matrix[start+r][j])||r-s).slice(0,2).forEach(r=>mask[start+r][j]=1);
 }
 let removed=0,count=0;
 $('perm-rows').innerHTML=matrix.map((row,i)=>'<tr'+(i===4?' style="border-top:3px solid var(--accent)"':'')+'><th>'+String(order[i]+1)+'</th>'+row.map((v,j)=>{
  if(v&&!mask[i][j]){removed+=v*v;count++;}
  return '<td'+(v&&mask[i][j]?' style="background:var(--accent-soft);color:var(--accent)"':'')+'>'+(v&&!mask[i][j]?'删 '+v:v)+'</td>';
 }).join('')+'</tr>').join('');
 $('perm-result').textContent='原有 16 个非零，平方能量 120。列方向新增删除 '+count+' 个，删除能量 '+removed+'（'+(removed/120*100).toFixed(0)+'%）。';
 $('perm-result').dataset.energy=removed;$('perm-result').dataset.count=count;
}
$('perm-order').addEventListener('change',renderPerm);
$('perm-reset').addEventListener('click',()=>{$('perm-order').value='original';renderPerm();});
renderThreshold();renderUnion();renderPerm();
})();
