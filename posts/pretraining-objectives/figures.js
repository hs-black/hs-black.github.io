(() => {
  'use strict';
  const modes = {
    ntp: {raw:['A','B','C','D','E','F'], transformed:['A','B','C','D','E','F'], input:['A','B','C','D','E'], labels:['B','C','D','E','F'], focus:2, name:'因果文本流', context:null, note:'位置 2 的输入 C 预测 D；可读取 A、B、C。五个有效标签贡献损失。'},
    t5: {raw:['A','B','C','D','E','F'], transformed:['〈s0〉','B','C','〈s1〉','E','〈s2〉'], input:['〈BOS〉','〈s0〉','B','C','〈s1〉','E'], labels:['〈s0〉','B','C','〈s1〉','E','〈s2〉'], focus:2, name:'待生成目标', context:['A','〈s0〉','D','〈s1〉','F'], note:'预测 C 时，decoder 已读 BOS、s0、B，并可读取整个 encoder 条件。六个目标位置参与计分。'},
    fim: {raw:['A','B','C','D','E','F'], transformed:['〈PRE〉','A','B','〈SUF〉','E','F','〈MID〉','C','D','〈EOS〉'], input:['〈PRE〉','A','B','〈SUF〉','E','F','〈MID〉','C','D'], labels:['A','B','〈SUF〉','E','F','〈MID〉','C','D','〈EOS〉'], focus:6, name:'重排后的流', context:null, note:'输入 MID 预测 C；左右条件 AB、EF 都在其因果前缀中。完整重排流有九个标签，其中 middle 内容为 C、D。'}
  };
  function row(label, tokens, active, target) {
    return '<div class="token-row"><span class="token-label">'+label+'</span><div class="token-track">'+tokens.map((t,i)=>'<span class="token '+(t.startsWith('〈')?'marker ':'')+(active && active(i)?(target?'target':'visible'):'')+'">'+t+'<small>'+i+'</small></span>').join('')+'</div></div>';
  }
  function drawObjective(_, stage) {
    const m=modes[document.getElementById('objective-mode').value];
    let html=row('原文',m.raw);
    if(stage>=1){if(m.context)html+=row('Encoder 条件',m.context,()=>stage===3);html+=row(m.name,m.transformed);}
    if(stage>=2){html+=row(m.context?'Decoder 输入':'模型输入',m.input,i=>stage===3&&i<=m.focus);html+=row('预测标签',m.labels,i=>stage===3&&i===m.focus,true);}
    document.getElementById('objective-rows').innerHTML=html;
    document.getElementById('objective-readout').textContent=stage===0?'先固定原始 token。接下来改变的是条件安排和监督目标。':stage===1?(m.context?'缺失片段被收集到目标流，encoder 保留两侧内容。':'训练流已经构造完成，接下来沿这条流建立因果预测。'):stage===2?'输入和标签相差一个位置。每个标签对应它左侧已经给定的内容。':m.note;
  }
  const objective = ArticleSteps.mount('#objective-flow',{duration:1800,steps:[{title:'原文',caption:'固定同一组内容 token，选择一种目标构造。'},{title:'构造训练流',caption:'特殊符号记录片段身份；T5 同时生成 encoder 条件。'},{title:'配对输入与标签',caption:'一行输入对应一行标签，预测位置与可见内容由右移和因果 mask 一起确定。'},{title:'检查一次真实预测',caption:'实心标签对应的输入前缀用边框标出，检查监督是否使用了预期条件。'}],render:drawObjective});
  document.getElementById('objective-mode').addEventListener('change',()=>{objective.pause();objective.render();});
  function drawMtp(_, anchor) {
    const horizon=+document.getElementById('mtp-horizon').value, split=document.getElementById('mtp-split').checked, letters='ABCDEFG';let count=0, selected=[];
    let html='<span class="pair-head">距离 ↓<br>位置 →</span>'+[...letters].map((x,i)=>'<span class="pair-head">'+i+' · '+x+'</span>').join('');
    for(let h=1;h<=horizon;h++){
      html+='<span class="pair-head">h='+h+'</span>';
      for(let t=0;t<7;t++){
        const valid=t+h<7&&(!split||(t<4)===(t+h<4));if(valid)count++;
        if(valid&&t===anchor)selected.push('h='+h+' → '+letters[t+h]);
        html+='<span class="pair-cell '+(split&&t===4?'boundary ':'')+(valid?(t===anchor?'selected':''):'invalid')+'">'+(valid?letters[t+h]:'—')+'</span>';
      }
    }
    document.getElementById('mtp-grid').innerHTML=html;
    document.getElementById('mtp-readout').textContent='有效监督总数 = '+count+'。当前表示读到 '+letters[anchor]+'；'+(selected.length?selected.join('，'):'位于文档末尾，没有后续标签。');
    document.getElementById('mtp-targets').dataset.validPairs=count;
  }
  const mtp=ArticleSteps.mount('#mtp-targets',{duration:1400,steps:[...'ABCDEFG'].map((x,i)=>({title:'共享表示 z'+i+' · 输入 '+x,caption:'当前列的所有有效头共用这个位置的主干表示；标签受距离和文档边界共同限制。'})),render:drawMtp});
  ['mtp-horizon','mtp-split'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{mtp.pause();mtp.render();}));
})();
