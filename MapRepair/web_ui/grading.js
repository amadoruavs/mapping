'use strict';
(() => {
  const el=id=>document.getElementById(id);
  const identity=()=>[[0,0],[1,1]];
  const defaults=()=>({master:identity(),red:identity(),green:identity(),blue:identity()});
  const specs=[],inputs=new Map();
  let job=null,values={},tone=defaults(),channel='master',selected=0,revision=0;
  let pending=false,rendering=false,exporting=false,previewURL=null,originalURL=null,compare=false;
  let drag=null,keyboardTimer=null,renderedRevision=-1,activeRevision=-1;
  let detailPosition=[.5,.5],detailURL=null,detailBusy=false,detailVersion=0;
  const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
  const save=()=>{if(job)try{localStorage.setItem('maprepair-grade-'+job,JSON.stringify(snapshot()));}catch(_){}};
  const snapshot=()=>({...values,curves:structuredClone(tone)});
  function section(title,target,open=false) {
    const details=document.createElement('details');details.className='grade-section';details.open=open;
    const heading=document.createElement('summary');heading.textContent=title;details.append(heading);el(target).append(details);return details;
  }
  function slider(parent,key,title,min=-100,max=100,step=1,value=0) {
    specs.push([key,min,max,step,value]);values[key]=value;
    const row=document.createElement('div');row.className='grade-row';
    const label=document.createElement('label');label.htmlFor='grade-'+key;label.textContent=title;
    const range=document.createElement('input');Object.assign(range,{id:'grade-'+key,type:'range',min,max,step,value});
    const number=document.createElement('input');Object.assign(number,{type:'number',min,max,step,value});number.setAttribute('aria-label',title+' value');
    const update=(source,commit=false)=>{
      const n=Number(source.value);if(source.value===''||!Number.isFinite(n))return;
      const next=clamp(n,min,max);if(values[key]!==next)revision++;values[key]=next;range.value=next;number.value=next;
      save();if(commit)schedule();
    };
    range.addEventListener('input',()=>update(range));
    range.addEventListener('change',()=>update(range,true));
    range.addEventListener('pointerup',()=>schedule());
    range.addEventListener('keyup',event=>{if(event.key.startsWith('Arrow')||['Home','End','PageUp','PageDown'].includes(event.key))schedule();});
    number.addEventListener('change',()=>update(number,true));
    number.addEventListener('keydown',event=>{if(event.key==='Enter'){update(number,true);number.blur();}});
    inputs.set(key,[range,number]);row.append(label,range,number);parent.append(row);
  }
  const light=section('Light & white balance','grade-controls',true);
  slider(light,'exposure','Exposure · stops',-3,3,.05);slider(light,'temperature','Temperature');slider(light,'tint','Tint');
  for(const [key,title] of [['contrast','Contrast'],['highlights','Highlights'],['shadows','Shadows'],['whites','Whites'],['blacks','Blacks']])slider(light,key,title);
  slider(light,'gamma','Gamma',.2,3,.02,1);
  const color=section('Color','grade-controls');
  slider(color,'saturation','Saturation');slider(color,'vibrance','Vibrance');slider(color,'hue','Hue rotation · °',-180,180);slider(color,'fade','Fade',0,100);
  const mixer=section('Color mixer · HSL','grade-extra-controls');
  for(const band of ['red','orange','yellow','green','aqua','blue','purple','magenta']){
    const d=document.createElement('details'),s=document.createElement('summary');s.textContent=band[0].toUpperCase()+band.slice(1);d.append(s);mixer.append(d);
    slider(d,band+'_hue','Hue · °',-60,60);slider(d,band+'_saturation','Saturation');slider(d,band+'_luminance','Luminance');
  }
  const balance=section('Color balance','grade-extra-controls');
  for(const [zone,title] of [['shadow','Shadows'],['midtone','Midtones'],['highlight','Highlights']]){
    const h=document.createElement('h3');h.textContent=title;balance.append(h);
    for(const c of ['red','green','blue'])slider(balance,zone+'_'+c,c[0].toUpperCase()+c.slice(1));
  }
  const detail=section('Detail & sharpening','grade-extra-controls');
  slider(detail,'clarity','Clarity');slider(detail,'texture','Texture');slider(detail,'sharpen','Sharpening',0,150);
  slider(detail,'radius','Radius · full-size px',.3,3,.1,1);slider(detail,'threshold','Threshold',0,20,1,3);
  slider(detail,'detail_protection','Halo protection',0,100,1,100);
  const note=document.createElement('p');note.className='hint';note.textContent='Detail radii scale with the preview. Fine sharpening is best judged in the full-size download.';detail.append(note);
  function refreshInputs(){for(const [key,, , ,def] of specs){values[key]??=def;for(const input of inputs.get(key))input.value=values[key];}}
  function schedule(){clearTimeout(keyboardTimer);if((rendering&&activeRevision===revision)||(renderedRevision===revision&&previewURL))return;pending=true;save();if(job&&!rendering)void preview();}
  async function request(settings,previewOnly,id,format='jpeg',crop=null){
    const response=await fetch(`/api/jobs/${id}/grade`,{method:'POST',headers:{'X-MapRepair':'1','Content-Type':'application/json'},body:JSON.stringify({settings,preview:previewOnly,format,crop})});
    if(!response.ok){const error=new Error((await response.json()).error||'Rendering failed.');error.status=response.status;throw error;}
    return response.blob();
  }
  async function preview(){
    if(!job||rendering||!pending)return;
    pending=false;rendering=true;activeRevision=revision;const id=job,version=revision,settings=snapshot(),start=performance.now();
    el('grade-status').textContent='Updating preview…';el('grade-image').setAttribute('aria-busy','true');
    try{
      const blob=await request(settings,true,id);
      if(id!==job||version!==revision)return;
      const url=URL.createObjectURL(blob);const loaded=new Image();loaded.src=url;
      try{await loaded.decode();}catch(error){URL.revokeObjectURL(url);throw error;}
      if(id!==job||version!==revision){URL.revokeObjectURL(url);return;}
      if(previewURL)URL.revokeObjectURL(previewURL);previewURL=url;renderedRevision=version;
      const neutral=specs.every(([key,,,,def])=>settings[key]===def)&&Object.values(settings.curves).every(points=>points.every(([x,y])=>x===y));
      if(neutral&&!originalURL)originalURL=URL.createObjectURL(blob);
      el('grade-image').src=compare&&originalURL?originalURL:previewURL;el('grade-image').hidden=false;
      el('grade-status').textContent=`${compare?'Original · ':''}${loaded.naturalWidth} × ${loaded.naturalHeight} preview · ${Math.round(performance.now()-start)} ms`;
    }catch(error){
      if(id===job&&version===revision){el('grade-status').textContent=error.message;if(error.status===409){pending=false;setTimeout(()=>{if(job===id)schedule();},200);}}
    }finally{
      rendering=false;el('grade-image').setAttribute('aria-busy','false');
      if(pending)requestAnimationFrame(()=>void preview());
      else if(el('grade-detail').open)void detailPreview();
    }
  }
  el('grade-export').onclick=async()=>{
    if(!job||exporting)return;exporting=true;el('grade-export').disabled=true;
    const id=job,settings=snapshot(),version=revision;el('grade-export-status').textContent='Rendering full resolution… You can keep editing.';
    try{
      const format=el('grade-format').value;
      const blob=await request(settings,false,id,format);if(id!==job)return;
      const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='maprepair_graded_full.'+(format==='png'?'png':'jpg');a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
      el('grade-export-status').textContent=revision===version?'Full-resolution grade downloaded.':'Downloaded the settings from when you clicked. Download again to include newer edits.';
    }catch(error){if(id===job)el('grade-export-status').textContent=error.message;}
    finally{exporting=false;el('grade-export').disabled=false;}
  };
  async function detailPreview(){
    if(!job||detailBusy||rendering||!el('grade-detail').open)return;
    detailBusy=true;const id=job,version=revision,positionVersion=detailVersion,comparison=compare;
    el('grade-detail-status').textContent='Rendering source detail…';
    try{
      const blob=await request(comparison?{}:snapshot(),true,id,'png',detailPosition);
      if(id!==job||version!==revision||positionVersion!==detailVersion||comparison!==compare)return;
      if(detailURL)URL.revokeObjectURL(detailURL);detailURL=URL.createObjectURL(blob);el('grade-detail-image').src=detailURL;
      el('grade-detail-status').textContent=(comparison?'Original':'Current grade')+' · full-resolution crop · lossless';
    }catch(error){if(id===job)el('grade-detail-status').textContent=error.message;}
    finally{detailBusy=false;if(id===job&&(version!==revision||positionVersion!==detailVersion||comparison!==compare)&&!rendering)void detailPreview();}
  }
  el('grade-detail').addEventListener('toggle',()=>{if(el('grade-detail').open)void detailPreview();});
  el('grade-image').addEventListener('click',event=>{
    const image=el('grade-image'),box=image.getBoundingClientRect(),scale=Math.min(box.width/image.naturalWidth,box.height/image.naturalHeight);
    const width=image.naturalWidth*scale,height=image.naturalHeight*scale;
    const x=(event.clientX-box.left-(box.width-width)/2)/width,y=(event.clientY-box.top-(box.height-height)/2)/height;
    if(x<0||x>1||y<0||y>1)return;detailPosition=[x,y];detailVersion++;el('grade-detail').open=true;void detailPreview();
  });
  el('grade-auto').onclick=async()=>{
    if(!job)return;
    const id=job,version=revision;el('grade-auto').disabled=true;el('grade-auto-status').textContent='Analyzing tones and fine-detail noise on this computer…';
    try{
      const response=await fetch(`/api/jobs/${id}/auto-grade`,{method:'POST',headers:{'X-MapRepair':'1'}});
      const result=await response.json();if(!response.ok)throw new Error(result.error||'Analysis failed.');
      if(id!==job)return;
      if(version!==revision){el('grade-auto-status').textContent='Your newer edits were kept. Click Auto cinema again to apply a fresh starting point.';return;}
      for(const [key,,,,def] of specs)values[key]=result.settings[key]??def;
      tone=result.settings.curves;selected=0;revision++;refreshInputs();drawCurve();schedule();
      el('grade-auto-status').textContent='Offline grade applied. Adjust any slider or curve to taste.'+(result.lossless_source?'':' This older result was saved as JPEG; PNG prevents further export compression.');
    }catch(error){if(id===job)el('grade-auto-status').textContent=error.message;}
    finally{el('grade-auto').disabled=false;}
  };
  el('grade-reset').onclick=()=>{for(const [key,,,,def] of specs)values[key]=def;tone=defaults();selected=0;revision++;refreshInputs();drawCurve();schedule();};
  el('grade-compare').onclick=async()=>{
    compare=!compare;el('grade-compare').setAttribute('aria-pressed',String(compare));el('grade-compare').textContent=compare?'Show grade':'Show original';
    if(compare&&!originalURL){
      const id=job;try{const blob=await request({},true,id);if(id!==job)return;originalURL=URL.createObjectURL(blob);}catch(error){el('grade-status').textContent=error.message;compare=false;el('grade-compare').textContent='Show original';el('grade-compare').setAttribute('aria-pressed','false');}
    }
    if(compare&&originalURL)el('grade-image').src=originalURL;else if(previewURL)el('grade-image').src=previewURL;
    void detailPreview();
  };
  // Same shape-preserving cubic interpolation used by the export renderer.
  function curveSamples(points){
    const n=points.length,h=[],d=[],m=Array(n).fill(0);
    for(let i=0;i<n-1;i++){h[i]=points[i+1][0]-points[i][0];d[i]=(points[i+1][1]-points[i][1])/h[i];}
    m[0]=d[0];m[n-1]=d[n-2];
    for(let i=1;i<n-1;i++)if(d[i-1]*d[i]>0){const a=2*h[i]+h[i-1],b=h[i]+2*h[i-1];m[i]=(a+b)/(a/d[i-1]+b/d[i]);}
    let j=0;return Array.from({length:257},(_,i)=>{const x=i/256;while(j<n-2&&x>points[j+1][0])j++;const t=(x-points[j][0])/h[j],t2=t*t,t3=t2*t;return [x,clamp((2*t3-3*t2+1)*points[j][1]+(t3-2*t2+t)*h[j]*m[j]+(-2*t3+3*t2)*points[j+1][1]+(t3-t2)*h[j]*m[j+1],0,1)];});
  }
  const svg=el('curve-editor'),ns='http://www.w3.org/2000/svg';
  function shape(tag,attributes){const node=document.createElementNS(ns,tag);for(const [key,v] of Object.entries(attributes))node.setAttribute(key,v);svg.append(node);return node;}
  function drawCurve(){
    const focus=document.activeElement?.dataset?.point;svg.replaceChildren();svg.dataset.channel=channel;
    for(const v of [0,64,128,192,256]){shape('line',{x1:v,y1:0,x2:v,y2:256,class:'curve-grid'});shape('line',{x1:0,y1:v,x2:256,y2:v,class:'curve-grid'});}
    shape('line',{x1:0,y1:256,x2:256,y2:0,class:'curve-identity'});
    shape('path',{d:curveSamples(tone[channel]).map(([x,y],i)=>`${i?'L':'M'} ${x*256} ${(1-y)*256}`).join(' '),class:'curve-line'});
    tone[channel].forEach(([x,y],i)=>{const point=shape('circle',{cx:x*256,cy:(1-y)*256,r:i===selected?6:4.5,class:'curve-point',tabindex:0,role:'button','data-point':i,'aria-label':`Point ${i+1}: input ${Math.round(x*255)}, output ${Math.round(y*255)}`,'aria-pressed':String(i===selected)});point.addEventListener('focus',()=>{selected=i;pointFields();});});
    if(focus!==undefined)svg.querySelector(`[data-point="${selected}"]`)?.focus({preventScroll:true});pointFields();
  }
  function pointFields(){const p=tone[channel];selected=clamp(selected,0,p.length-1);el('curve-input').value=Math.round(p[selected][0]*255);el('curve-output').value=Math.round(p[selected][1]*255);el('curve-input').disabled=selected===0||selected===p.length-1;el('curve-remove').disabled=selected===0||selected===p.length-1;el('curve-add').disabled=p.length>=16;}
  function position(event){const box=svg.getBoundingClientRect();return [clamp((event.clientX-box.left)/box.width,0,1),clamp(1-(event.clientY-box.top)/box.height,0,1)];}
  function move(x,y){const p=tone[channel];if(selected>0&&selected<p.length-1)p[selected][0]=clamp(x,p[selected-1][0]+1/255,p[selected+1][0]-1/255);p[selected][1]=clamp(y,0,1);revision++;el('curve-preset').value='';drawCurve();}
  function add(x,y){const p=tone[channel];if(p.length>=16)return false;const i=p.findIndex(pt=>pt[0]>x);if(i<1||x-p[i-1][0]<1/255||p[i][0]-x<1/255)return false;p.splice(i,0,[x,y]);selected=i;revision++;drawCurve();return true;}
  function remove(){const p=tone[channel];if(selected===0||selected===p.length-1)return;p.splice(selected,1);selected--;revision++;drawCurve();schedule();}
  svg.addEventListener('pointerdown',event=>{if(event.button!==0)return;const [x,y]=position(event);if(event.target.dataset.point!==undefined)selected=Number(event.target.dataset.point);else if(!add(x,y))return;event.preventDefault();drag=event.pointerId;svg.setPointerCapture(drag);drawCurve();});
  svg.addEventListener('pointermove',event=>{if(drag===event.pointerId)move(...position(event));});
  const release=event=>{if(drag!==event.pointerId)return;drag=null;schedule();};svg.addEventListener('pointerup',release);svg.addEventListener('pointercancel',release);
  svg.addEventListener('keydown',event=>{const p=tone[channel][selected],step=(event.shiftKey?10:1)/255;if(['Delete','Backspace'].includes(event.key)){event.preventDefault();remove();}else if(event.key.startsWith('Arrow')){event.preventDefault();move(p[0]+(event.key==='ArrowRight'?step:event.key==='ArrowLeft'?-step:0),p[1]+(event.key==='ArrowUp'?step:event.key==='ArrowDown'?-step:0));clearTimeout(keyboardTimer);keyboardTimer=setTimeout(schedule,100);}});
  for(const id of ['curve-input','curve-output'])el(id).addEventListener('change',()=>{const x=Number(el('curve-input').value),y=Number(el('curve-output').value);if(Number.isFinite(x)&&Number.isFinite(y)){move(x/255,y/255);schedule();}});
  el('curve-add').onclick=()=>{const p=tone[channel];let i=0;for(let j=1;j<p.length-1;j++)if(p[j+1][0]-p[j][0]>p[i+1][0]-p[i][0])i=j;if(add((p[i][0]+p[i+1][0])/2,(p[i][1]+p[i+1][1])/2))schedule();};
  el('curve-remove').onclick=remove;
  el('curve-channel').onchange=()=>{channel=el('curve-channel').value;selected=0;el('curve-preset').value='';drawCurve();};
  el('curve-preset').onchange=()=>{const presets={linear:identity(),soft:[[0,0],[.25,.19],[.75,.81],[1,1]],strong:[[0,0],[.25,.12],[.75,.88],[1,1]],matte:[[0,.08],[.25,.25],[.75,.78],[1,.96]]};const p=presets[el('curve-preset').value];if(p){tone[channel]=p;selected=0;revision++;drawCurve();schedule();}};
  window.MapGrade={setJob(state){
    if(state.state!=='complete'){if(job){job=null;revision++;}return;}
    if(job===state.id)return;job=state.id;revision++;compare=false;detailPosition=[.5,.5];detailVersion++;
    el('grade-detail').open=false;el('grade-detail-image').removeAttribute('src');if(detailURL)URL.revokeObjectURL(detailURL);detailURL=null;
    el('grade-compare').textContent='Show original';el('grade-compare').setAttribute('aria-pressed','false');
    for(const url of [previewURL,originalURL])if(url)URL.revokeObjectURL(url);previewURL=originalURL=null;
    for(const [key,,,,def] of specs)values[key]=def;tone=defaults();
    try{const stored=JSON.parse(localStorage.getItem('maprepair-grade-'+job));if(stored){for(const [key,min,max] of specs)if(Number.isFinite(stored[key]))values[key]=clamp(stored[key],min,max);if(stored.curves)for(const key of Object.keys(tone)){const p=stored.curves[key];if(Array.isArray(p)&&p.length>=2&&p.length<=16&&p.every((v,i)=>Array.isArray(v)&&v.length===2&&v.every(n=>Number.isFinite(n)&&n>=0&&n<=1)&&(!i||v[0]-p[i-1][0]>=1/1024))&&p[0][0]===0&&p.at(-1)[0]===1)tone[key]=p;}}}catch(_){}
    refreshInputs();drawCurve();el('grade-image').src=state.preview;el('grade-image').hidden=false;el('preview').hidden=true;
    document.querySelector('main').classList.add('grading-ready');el('grade-export-status').textContent='';schedule();
  }};
  drawCurve();if(window.currentGradeJob)window.MapGrade.setJob(window.currentGradeJob);
})();
