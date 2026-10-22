(() => {
'use strict';
const $ = id => document.getElementById(id);
const weights = [.9,.8,.7,.6,.5,.4,.3,.2], scale = [1,3,.5,2];
function renderMask() {
  const score = weights.map((w,i) => Math.abs(w) * ($('mask-score').value === 'wanda' ? scale[i%4] : 1));
  const rank = ids => ids.sort((a,b) => score[b]-score[a] || a-b);
  const kept = $('mask-pattern').value === '24'
    ? [...rank([0,1,2,3]).slice(0,2),...rank([4,5,6,7]).slice(0,2)]
    : rank(weights.map((_,i)=>i)).slice(0,4);
  const mask = new Set(kept);
  $('mask-cells').innerHTML = weights.map((w,i) => '<div class="s-cell '+(mask.has(i)?'on':'dim')+'">w = '+w.toFixed(1)+'<small>分数 '+score[i].toFixed(2)+' · '+(mask.has(i)?'保留':'移除')+'</small></div>').join('');
  const error = weights.reduce((v,w,i)=>v+(mask.has(i)?0:w*w*scale[i%4]**2),0);
  $('mask-result').textContent = '保留 4 / 8 个权重；两行分别保留 '+[0,1].map(r=>kept.filter(i=>Math.floor(i/4)===r).length).join('、')+' 个。独立输入下，两路输出的均方误差之和 = '+error.toFixed(4)+'。';
  $('mask-result').dataset.error=error.toFixed(8);
}
['mask-score','mask-pattern'].forEach(id=>$(id).addEventListener('change',renderMask));
$('mask-reset').addEventListener('click',()=>{$('mask-score').value='magnitude';$('mask-pattern').value='unstructured';renderMask();});
function renderComp() {
  const rho=Number($('comp-rho').value), x1=[1,-1,1,-1], z=[1,1,-1,-1];
  let error=0;
  $('comp-rho-value').textContent=rho.toFixed(1);
  $('comp-rows').innerHTML=x1.map((a,i)=>{
    const b=rho*a+Math.sqrt(1-rho*rho)*z[i], y=a+b, after=(1+rho)*b;
    error+=(y-after)**2/4;
    return '<tr><td>'+(i+1)+'</td><td>'+y.toFixed(3)+'</td><td>'+b.toFixed(3)+'</td><td>'+after.toFixed(3)+'</td></tr>';
  }).join('');
  $('comp-error-label').textContent='补偿后误差 '+error.toFixed(3);
  $('comp-error-bar').style.width=(error*100)+'%';
  $('comp-result').textContent='保留权重从 1 调整为 '+(1+rho).toFixed(2)+'。直接删除的均方误差 = 1；补偿后 = '+error.toFixed(3)+'，与 1 − ρ² 一致。';
  $('comp-result').dataset.error=error.toFixed(8);
}
$('comp-rho').addEventListener('input',renderComp);
$('comp-reset').addEventListener('click',()=>{$('comp-rho').value='.8';renderComp();});
renderMask();renderComp();
})();
