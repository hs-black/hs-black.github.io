(() => {
  'use strict';
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0), norm=x=>Math.sqrt(dot(x,x)), mv=(m,x)=>m.map(r=>dot(r,x));
  const transpose=m=>m[0].map((_,i)=>m.map(r=>r[i]));
  const eye=()=>Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>+(i===j)));
  const plus=(a,b)=>a.map((r,i)=>r.map((v,j)=>v+b[i][j]));
  const mm=(a,b)=>a.map(r=>transpose(b).map(c=>dot(r,c)));
  function rms(x){const r=Math.sqrt(dot(x,x)/4+1e-4);return {value:x.map(v=>v/r),jac:eye().map((row,i)=>row.map((v,j)=>v/r-x[i]*x[j]/(4*r*r*r))) };}
  function probe(){
    const mode=document.getElementById('norm-placement').value, scale=+document.getElementById('branch-scale').value;
    const xs=[[.3,-.9,1.2,.6]], js=[];
    for(let l=0;l<6;l++){
      const w=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>scale*Math.sin((l+1)*(i+1)+(j+1)*1.7)/2)),x=xs[l];
      if(mode==='pre'){const n=rms(x),f=mv(w,n.value);xs.push(x.map((v,i)=>v+f[i]));js.push(plus(eye(),mm(w,n.jac)));}
      else{const f=mv(w,x),n=rms(x.map((v,i)=>v+f[i]));xs.push(n.value);js.push(mm(n.jac,plus(eye(),w)));}
    }
    const gs=Array(7);gs[6]=[1,-.5,.2,.7];for(let l=5;l>=0;l--)gs[l]=mv(transpose(js[l]),gs[l+1]);
    return {xs,gs,forward:xs.map(v=>norm(v)/2),backward:gs.map(norm)};
  }
  function drawGradient(_,step){
    const p=probe(),active=6-step,mf=Math.max(...p.forward),mg=Math.max(...p.backward);let html='';
    for(let l=0;l<=6;l++){
      const known=l>=active;
      html+='<div class="gradient-row '+(l===active?'active':'')+'"><span>'+l+'</span><div class="gradient-value"><div class="gradient-track"><div class="gradient-fill forward" style="width:'+p.forward[l]/mf*100+'%"></div></div><output>'+p.forward[l].toFixed(3)+'</output></div><div class="gradient-value"><div class="gradient-track"><div class="gradient-fill" style="width:'+(known?p.backward[l]/mg*100:0)+'%"></div></div><output>'+(known?p.backward[l].toFixed(3):'待算')+'</output></div></div>';
    }
    document.getElementById('gradient-rows').innerHTML=html;
    document.getElementById('gradient-readout').textContent='当前 g'+active+' = ['+p.gs[active].map(v=>v.toFixed(3)).join(', ')+']；范数 '+p.backward[active].toFixed(4)+'。'+(step?'由 g'+(active+1)+' 乘第 '+active+' 层 Jacobian 的转置得到。':'这是给定的顶层梯度，下一步向第 5 层反传。');
    document.getElementById('gradient-flow').dataset.inputGradient=p.backward[0];
  }
  const gradient=ArticleSteps.mount('#gradient-flow',{duration:1400,steps:Array.from({length:7},(_,i)=>({title:i?'反传到第 '+(6-i)+' 层':'固定顶层梯度',caption:i?'前向状态保持固定，局部导数按前向实际数值求值。每次只跨过一层。':'先计算六层前向，再从顶层给定方向开始逐层反传。'})),render:drawGradient});
  ['norm-placement','branch-scale'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{gradient.pause();gradient.render();}));
  function drawEntropy(_,step){
    const a=+document.getElementById('attention-scale').value,z=[1,0,-1],logits=z.map(x=>a*x),exp=logits.map(x=>Math.exp(x-Math.max(...logits))),total=exp.reduce((s,x)=>s+x,0),p=exp.map(x=>x/total),mean=dot(p,z),h=-p.reduce((s,x)=>s+x*Math.log(x),0),dh=-a*dot(p,z.map(x=>(x-mean)**2));
    document.getElementById('attention-scale-value').textContent=a.toFixed(1);
    document.getElementById('entropy-bars').innerHTML=z.map((x,i)=>'<div class="score-row"><span>'+i+'</span><span>'+logits[i].toFixed(1)+'</span><div class="score-track"><div class="score-fill" style="width:'+p[i]*100+'%"></div></div><span>'+p[i].toFixed(3)+'</span></div>').join('');
    document.getElementById('entropy-value').textContent=h.toFixed(4);document.getElementById('max-probability').textContent=Math.max(...p).toFixed(4);document.getElementById('entropy-derivative').textContent=dh.toFixed(4);
    const messages=['基础 z = [1, 0, −1]，乘尺度后分数为 ['+logits.map(x=>x.toFixed(1)).join(', ')+']。概率条始终显示当前尺度的结果。','先减去最大分数 '+a.toFixed(1)+' 再取指数：['+exp.map(x=>x.toFixed(4)).join(', ')+']。共同平移保持 softmax 概率不变。','指数总和为 '+total.toFixed(4)+'；每项除以它得到三个概率，其和为 '+p.reduce((s,x)=>s+x,0).toFixed(4)+'。','每项贡献 −p log p，合计熵 '+h.toFixed(4)+' nats；固定分数方向时，导数 −a·Var(z) = '+dh.toFixed(4)+'。'];
    document.getElementById('entropy-readout').textContent=messages[step];document.getElementById('entropy-flow').dataset.entropy=h;
  }
  const entropy=ArticleSteps.mount('#entropy-flow',{duration:1800,steps:[{title:'缩放分数',caption:'尺度只改变分数差的大小，三个位置的次序保持不变。'},{title:'稳定地取指数',caption:'先减去最大分数，避免大正数直接取指数。'},{title:'归一化',caption:'所有指数除以同一个总和，得到概率。'},{title:'求和得到熵',caption:'熵把三个概率的集中程度汇总成一个数。'}],render:drawEntropy});
  document.getElementById('attention-scale').addEventListener('input',()=>{entropy.pause();entropy.render();});
})();
