(() => {
 'use strict';
 const $=id=>document.getElementById(id);
 const mean=ArticleSteps.mount('#mean-demo',{
  duration:1400,
  steps:[
   {title:'两个可能的目标',caption:'给定同一段前文，目标向量的概率质量位于 −2 和 +2。'},
   {title:'用平方误差选择预测',caption:'最优单点预测是条件均值：−2×(1−π)+2×π。'},
   {title:'计算剩余不确定性',caption:'最小 MSE 为 16π(1−π)；分布越均衡，单点预测的误差越大。'},
   {title:'再给一个带噪观测',caption:'去噪器还看到 y，先验权重更新成后验权重；它在更具体的条件下预测均值。'}
  ],
  render(step,index){
   const prior=+$('mean-prior').value,y=+$('mean-noisy').value;
   const p=index===3?1/(1+Math.exp(-(Math.log(prior/(1-prior))+4*Math.SQRT1_2*y/.5))):prior;
   const mu=4*p-2,risk=16*p*(1-p);
   $('mean-prior-out').textContent=prior.toFixed(1);$('mean-noisy-out').textContent=y.toFixed(1);
   $('mean-left-bar').style.width=(1-p)*100+'%';$('mean-right-bar').style.width=p*100+'%';
   $('mean-left').textContent=((1-p)*100).toFixed(2)+'%';$('mean-right').textContent=(p*100).toFixed(2)+'%';
   $('mean-dot').style.left=((mu+2)/4*100)+'%';$('mean-dot').style.visibility=index===0?'hidden':'visible';
   $('mean-estimate').textContent=index===0?'待计算':mu.toFixed(4);
   $('mean-risk').textContent=index<2?'待计算':risk.toFixed(4);
   $('mean-conditioning').textContent=index===3?'当前条件：前文 + 带噪观测 y='+y.toFixed(1)+'。两端数值是后验概率。':'当前条件：只有前文。两端数值是先验概率。';
  }
 });
 ['mean-prior','mean-noisy'].forEach(id=>$(id).addEventListener('input',()=>mean.render()));
 const probabilities=[1,.1,.2,.8,.1,.75,.1,.1],values=[2,4,6,8,10,12,14,16];
 function segments(threshold){
  const starts=probabilities.flatMap((p,i)=>p>=threshold?[i+1]:[]),rows=[];
  starts.forEach((start,i)=>{const end=i+1<starts.length?starts[i+1]-1:8,available=end+1;
   const members=values.slice(start-1,end);rows.push({start,end,available,mean:members.reduce((a,b)=>a+b,0)/members.length});});
  return rows;
 }
 const segment=ArticleSteps.mount('#segment-demo',{
  duration:1000,
  steps:Array.from({length:10},(_,t)=>({title:t===0?'还未读入 token':t===9?'读到文档结束标记':'已读到位置 '+t,caption:t===0?'先检查分段规则，随后逐个位置读入表示。':t===9?'结束标记闭合最后一段，全部完整段表示都已可用。':'边界在当前位置出现时，上一段闭合；当前新段继续沿 token 路径处理。'})),
  render(step,t){
   const threshold=+$('segment-threshold').value,rows=segments(threshold);
   $('segment-tokens').innerHTML=values.map((v,i)=>'<div class="cm-token'+(i+1<=t?' seen':'')+(probabilities[i]>=threshold?' boundary':'')+'"><span>位置 '+(i+1)+'</span><b>'+(i+1<=t?v:'待读')+'</b><small>p='+probabilities[i]+'</small></div>').join('');
   $('segment-time').textContent=t===9?'9 · 结束':String(t);
   const visible=rows.filter(r=>r.available<=t);
   $('segment-visible').textContent=visible.length?visible.map(r=>r.mean.toFixed(1)).join(' / '):'尚无';
   $('segment-rows').innerHTML=rows.map((r,k)=>'<div class="cm-segment'+(r.available<=t?' available':'')+'"><b>概念 '+(k+1)+' · 位置 '+r.start+'–'+r.end+'</b><small>可用时间 a='+r.available+'；'+(r.available<=t?'闭合均值 = '+r.mean.toFixed(1):'等待闭合')+'</small></div>').join('');
  }
 });
 $('segment-threshold').addEventListener('change',()=>segment.render());
 const copies=ArticleSteps.mount('#copies-demo',{
  duration:1500,
  steps:[
   {title:'每个概念出现一次',caption:'未归一化质量为 1 和 3，归一化后是 1/4 和 3/4。'},
   {title:'复制第二个概念',caption:'每份副本各自参与 softmax，第二组的总质量随可见副本数增加。'},
   {title:'按可见副本数补偿',caption:'第二组每份分数减去 log m，所有副本合计质量恢复为 3。'}
  ],
  render(step,index){
   const m=+$('copies-count').value,visible=index===0?1:m,second=index===1?3*m:3;
   const p1=1/(1+second),p2=1-p1,result=2*p1+10*p2;
   $('copies-count-out').textContent=String(m);
   $('copies-items').innerHTML='<span class="cm-copy">C₁ × 1</span>'+Array.from({length:visible},(_,j)=>'<span class="cm-copy second">C₂ · '+(j+1)+'</span>').join('');
   $('copies-bar1').style.width=p1*100+'%';$('copies-bar2').style.width=p2*100+'%';
   $('copies-weight1').textContent=p1.toFixed(4);$('copies-weight2').textContent=p2.toFixed(4);
   $('copies-output').textContent=result.toFixed(4);
   $('copies-formula').textContent=index===0?'输出 = (1×2 + 3×10) / 4 = 8':index===1?'第二组总质量 = '+m+'×3 = '+3*m+'；输出 = (2 + '+(30*m)+') / '+(1+3*m):'第二组总质量 = '+m+'×exp(ln 3−ln '+m+') = 3；输出恢复为 8。';
  }
 });
 $('copies-count').addEventListener('input',()=>copies.render());
 window.ConceptDemos={mean,segment,copies};
})();
