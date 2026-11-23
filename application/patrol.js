'use strict';
const $ = id => document.getElementById(id);
const state = {catalog:null, split:'val', model:'baseline', mode:'benchmark', selected:null, filtered:[], shown:60,
  images:null, upload:null, generation:0, swipe:false};
const groups = {empty:'无目标变化',small_0_to_1pct:'小变化',medium_1_to_10pct:'中等变化',large_over_10pct:'大变化'};
const percent = n => `${(100*n).toFixed(2)}%`;
const modelName = () => state.model==='baseline'?'基础 U-Net':'U-Net + 自注意力';
function notice(text,error=false){$('notice').textContent=text;$('notice').classList.toggle('error',error);}
function imageUrl(kind,name,model=state.model){return `/api/image?kind=${kind}&sample=${encodeURIComponent(name)}&model=${model}&split=${state.split}`;}
function loadImage(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('影像读取失败'));im.src=src;});}
function modelMetadata(){const meta=state.catalog.models[state.model];$('epochBadge').textContent=`验证选模权重 · 第 ${meta.epoch} 轮`;
  if(state.mode==='upload'){$('provenance').textContent=`${modelName()} · seed=42 · 第 ${meta.epoch} 轮 · 权重 ${meta.sha256.slice(0,12)}。当前输入没有参考标签，不计算该影像精度。`;return;}
  $('provenance').textContent=`${modelName()} · seed=42 · 第 ${meta.epoch} 轮 · 权重 ${meta.sha256.slice(0,12)} · 完整${state.split==='test'?'测试':'验证'}前景平均 IoU ${percent(meta.foreground_miou)}。训练已完成；测试结果不用于重新选模。`;}
function metrics(){if(state.mode==='upload')return state.upload?.result||null;return state.selected?.models[state.model]||null;}
function statistics(){const m=metrics();$('exportCsv').disabled=state.mode==='upload'&&!m;if(!m){for(const id of ['totalStat','roadStat','buildingStat','truthStat'])$(id).textContent='—';$('roadDetail').textContent='—';$('buildingDetail').textContent='—';return;}
  const upload=state.mode==='upload',road=upload?m.road_fraction:m.pred_road_fraction,building=upload?m.building_fraction:m.pred_building_fraction;
  $('totalStat').textContent=percent(road+building);$('roadStat').textContent=percent(road);$('buildingStat').textContent=percent(building);
  $('roadDetail').textContent=upload?`${m.road_pixels.toLocaleString()} 像素`:`参考 ${percent(m.true_road_fraction)} · 误差 ${m.road_area_abs_error_pp.toFixed(2)} 个百分点`;
  $('buildingDetail').textContent=upload?`${m.building_pixels.toLocaleString()} 像素`:`参考 ${percent(m.true_building_fraction)} · 误差 ${m.building_area_abs_error_pp.toFixed(2)} 个百分点`;
  $('fourthLabel').textContent=upload?'本次 CPU 前向耗时':'参考总变化占比';
  $('truthStat').textContent=upload?`${m.cpu_forward_seconds.toFixed(2)} s`:percent(m.true_change_pixels/m.pixels);
  $('truthDetail').textContent=upload?'单次预测，不含启动及权重加载':'基于数据集参考标签';
}
function filterSamples(){const m=state.model,query=$('search').value.trim().toLowerCase(),group=$('group').value,sort=$('sort').value;
  const ranked=[...state.catalog.samples].sort((a,b)=>b.models[m].pred_change_pixels/b.models[m].pixels-a.models[m].pred_change_pixels/a.models[m].pixels||a.name.localeCompare(b.name));
  const priority=new Set(ranked.slice(0,Math.ceil(.2*ranked.length)).map(s=>s.name));
  let rows=ranked.filter(s=>(!query||s.name.includes(query))&&(group==='all'||s.group===group)&&(!$('topOnly').checked||priority.has(s.name)));
  if(sort==='asc')rows.sort((a,b)=>a.models[m].pred_change_pixels/a.models[m].pixels-b.models[m].pred_change_pixels/b.models[m].pixels||a.name.localeCompare(b.name));
  if(sort==='name')rows.sort((a,b)=>a.name.localeCompare(b.name));
  state.filtered=rows;state.shown=60;$('count').textContent=rows.length;renderList();
  if(!rows.some(s=>s.name===state.selected?.name))selectSample(rows[0]||null);}
function renderList(){const list=$('sampleList'),scroll=list.scrollTop;list.replaceChildren();
  if(!state.filtered.length){const p=document.createElement('p');p.className='empty-list';p.textContent='没有匹配影像，请调整筛选条件。';list.append(p);return;}
  for(const s of state.filtered.slice(0,state.shown)){const button=document.createElement('button');button.className='sample-button';button.classList.toggle('active',s.name===state.selected?.name);button.setAttribute('aria-pressed',String(s.name===state.selected?.name));button.dataset.name=s.name;
    const img=document.createElement('img');img.loading='lazy';img.src=imageUrl('after',s.name);img.alt='';
    const title=document.createElement('span'),name=document.createElement('strong'),description=document.createElement('small');name.textContent=s.name.replace('.png','');description.textContent=groups[s.group];title.append(name,description);
    const fraction=document.createElement('span');fraction.className='fraction';fraction.textContent=percent(s.models[state.model].pred_change_pixels/s.models[state.model].pixels);
    button.append(img,title,fraction);button.onclick=()=>selectSample(s);list.append(button);}
  if(state.shown<state.filtered.length){const more=document.createElement('button');more.className='more-note';more.textContent=`载入更多（已显示 ${state.shown} / ${state.filtered.length}）`;more.onclick=()=>{state.shown+=60;renderList();};list.append(more);}list.scrollTop=scroll;}
function clearViewer(){state.images=null;for(const id of ['beforeCanvas','afterCanvas','overlayCanvas','swipeCanvas']){const c=$(id);c.getContext('2d').clearRect(0,0,c.width,c.height);}$('exportMask').disabled=true;statistics();}
async function selectSample(sample){state.selected=sample;renderList();const generation=++state.generation;
  if(!sample){$('sampleTitle').textContent='未选择影像';$('sampleSubtitle').textContent='调整左侧筛选条件';clearViewer();return;}
  $('sampleTitle').textContent=sample.name.replace('.png','');$('sampleSubtitle').textContent=`${state.split==='test'?'测试':'验证'}样本 / ${groups[sample.group]} / 256×256 像素`;
  clearViewer();notice('正在读取影像与已核验预测…');
  try{const [before,after,truth,prediction]=await Promise.all(['before','after','truth','prediction'].map(k=>loadImage(imageUrl(k,sample.name))));
    if(generation!==state.generation||state.mode!=='benchmark')return;state.images={before,after,truth,prediction};draw();statistics();$('exportMask').disabled=false;
    notice('基准影像演示 · 当前显示固定权重的预测；标签不覆盖所有地物变化。');
  }catch(error){if(generation===state.generation)notice(error.message,true);}}
function pixels(image){const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d');x.drawImage(image,0,0,256,256);return x.getImageData(0,0,256,256).data;}
function classAt(data,i){return data[i]>200?(data[i+1]>200?1:2):0;}
function active(c){return c===1?$('roadLayer').checked:c===2?$('buildingLayer').checked:false;}
function overlay(){const result=document.createElement('canvas');result.width=result.height=256;const ctx=result.getContext('2d'),data=ctx.createImageData(256,256),images=state.images,mode=$('layerMode').value;
  if(!images?.prediction)return result;const p=pixels(images.prediction),t=images.truth?pixels(images.truth):null;
  for(let i=0;i<p.length;i+=4){const pc=classAt(p,i),tc=t?classAt(t,i):0;let color=null;
    if(mode==='prediction'&&active(pc))color=pc===1?[255,232,0]:[255,76,65];
    if(mode==='truth'&&t&&active(tc))color=tc===1?[255,232,0]:[255,76,65];
    if(mode==='error'&&t){const pa=active(pc),ta=active(tc);if(pa&&ta)color=pc===tc?[78,201,131]:[185,117,239];else if(pa)color=[255,157,63];else if(ta)color=[63,198,235];}
    if(color){data.data[i]=color[0];data.data[i+1]=color[1];data.data[i+2]=color[2];data.data[i+3]=Math.round(255*Number($('opacity').value)/100);}}
  ctx.putImageData(data,0,0);return result;}
function draw(){if(!state.images)return;const {before,after}=state.images;for(const [id,image] of [['beforeCanvas',before],['afterCanvas',after],['overlayCanvas',after]]){const c=$(id).getContext('2d');c.clearRect(0,0,256,256);c.drawImage(image,0,0,256,256);}
  const layer=overlay();$('overlayCanvas').getContext('2d').drawImage(layer,0,0);
  const swipe=$('swipeCanvas').getContext('2d'),position=Number($('swipePosition').value);swipe.clearRect(0,0,256,256);swipe.drawImage(after,0,0,256,256);swipe.drawImage(layer,0,0);swipe.save();swipe.beginPath();swipe.rect(0,0,position,256);swipe.clip();swipe.drawImage(before,0,0,256,256);swipe.restore();swipe.strokeStyle='#fff';swipe.lineWidth=1;swipe.beginPath();swipe.moveTo(position,0);swipe.lineTo(position,256);swipe.stroke();
  const mode=$('layerMode').value;$('overlayTitle').textContent={prediction:'后期影像＋预测',truth:'后期影像＋参考标签',error:'后期影像＋逐类错误'}[mode];
  $('legend').innerHTML=mode==='error'?'<span><i class="dot" style="background:#4ec983"></i>正确命中</span><span><i class="dot" style="background:#ff9d3f"></i>误报</span><span><i class="dot" style="background:#3fc6eb"></i>漏检</span><span><i class="dot" style="background:#b975ef"></i>类别混淆（误报＋漏检）</span>':'<span><i class="dot road"></i>道路变化</span><span><i class="dot building"></i>建筑变化</span>';}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportMask(){if(!state.images?.prediction)return;const c=document.createElement('canvas');c.width=c.height=256;c.getContext('2d').drawImage(state.images.prediction,0,0,256,256);c.toBlob(blob=>{if(blob)download(blob,`${state.mode==='benchmark'?state.selected.name.replace('.png',''):'local_pair'}_${state.model}_ep${state.catalog.models[state.model].epoch}_prediction.png`);});}
function exportCsv(){const meta=state.catalog.models[state.model],headers=['filename','model','checkpoint_epoch','pred_road_fraction','pred_building_fraction','pred_change_fraction','true_road_fraction','true_building_fraction','road_area_abs_error_pp','building_area_abs_error_pp'];const rows=state.filtered.map(s=>{const r=s.models[state.model];return [s.name,state.model,meta.epoch,r.pred_road_fraction,r.pred_building_fraction,r.pred_change_pixels/r.pixels,r.true_road_fraction,r.true_building_fraction,r.road_area_abs_error_pp,r.building_area_abs_error_pp];});
  download(new Blob(['\ufeff'+[headers,...rows].map(r=>r.join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),`${state.split}_${state.model}_ep${meta.epoch}_filtered.csv`);}
function exportLocalStatistics(){const r=state.upload?.result;if(!r)return;const value={model:state.model,checkpoint_epoch:r.epoch,checkpoint_sha256:r.provenance.sha256,input_width:256,input_height:256,reference_available:false,road_pixels:r.road_pixels,building_pixels:r.building_pixels,road_fraction:r.road_fraction,building_fraction:r.building_fraction,total_change_fraction:r.road_fraction+r.building_fraction,cpu_forward_seconds:r.cpu_forward_seconds,note:r.note};download(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}),`local_pair_${state.model}_ep${r.epoch}_statistics.json`);}
function setMode(mode){++state.generation;state.mode=mode;$('split').disabled=mode==='upload';for(const [id,m] of [['benchmarkTab','benchmark'],['uploadTab','upload']]){$(id).classList.toggle('active',m===mode);$(id).setAttribute('aria-pressed',String(m===mode));}
  $('samplePanel').hidden=mode!=='benchmark';$('uploadPanel').hidden=mode!=='upload';$('exportCsv').textContent=mode==='upload'?'导出预测统计 JSON ↓':'导出筛选列表 CSV ↓';if(state.catalog)modelMetadata();
  for(const option of $('layerMode').options)option.disabled=mode==='upload'&&option.value!=='prediction';$('layerMode').value='prediction';
  if(mode==='benchmark'){modelMetadata();selectSample(state.selected||state.filtered[0]);}
  else{state.selected=state.selected||state.filtered[0];state.upload=null;clearViewer();$('sampleTitle').textContent='本地影像预测';$('sampleSubtitle').textContent='选择前后两期 RGB 影像，在本机生成变化掩膜';$('fourthLabel').textContent='本次 CPU 前向耗时';$('truthDetail').textContent='无参考标签，不计算精度';notice('请在左侧选择同一区域的两期 256×256 RGB 影像。');prepareUpload();}}
function readFile(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error('文件读取失败'));reader.readAsDataURL(file);});}
async function prepareUpload(){const a=$('beforeFile').files[0],b=$('afterFile').files[0];if(!a||!b)return;const generation=++state.generation;state.upload=null;clearViewer();
  try{const data=await Promise.all([readFile(a),readFile(b)]),images=await Promise.all(data.map(loadImage));if(generation!==state.generation||state.mode!=='upload')return;
    if(images.some(im=>im.naturalWidth!==256||im.naturalHeight!==256))throw new Error('影像必须为 256×256 像素；页面不会自动缩放后送入模型。');
    state.upload={data,images,result:null};state.images={before:images[0],after:images[1],prediction:null,truth:null};draw();notice('两期影像已载入。确认顺序与配准后，点击“预测变化区域”。');
  }catch(error){notice(error.message,true);}}
async function predict(event){event.preventDefault();if(!state.upload){notice('请先选择两张有效的 256×256 影像。',true);return;}
  const button=$('predictButton'),model=state.model,generation=++state.generation;button.disabled=true;button.textContent='正在本机预测…';notice('正在用 CPU 运行所选模型，请稍候…');
  try{const data=state.upload.data,response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,checkpoint_sha256:state.catalog.models[model].sha256,before:data[0].split(',')[1],after:data[1].split(',')[1]})}),result=await response.json();if(!response.ok)throw new Error(result.error||'预测失败');
    const prediction=await loadImage(`data:image/png;base64,${result.mask_rgb}`);if(generation!==state.generation||state.mode!=='upload'||model!==state.model)return;
    state.upload.result=result;state.images.prediction=prediction;draw();statistics();$('exportMask').disabled=false;notice(`预测完成 · ${modelName()}第 ${result.epoch} 轮权重 · 无参考标签，未计算精度。`);
  }catch(error){if(generation===state.generation)notice(error.message,true);}finally{button.disabled=false;button.textContent='预测变化区域';}}
$('model').onchange=()=>{state.model=$('model').value;modelMetadata();if(state.mode==='benchmark'){filterSamples();selectSample(state.selected);}else if(state.upload){++state.generation;state.upload.result=null;state.images.prediction=null;draw();statistics();$('exportMask').disabled=true;notice('已切换模型，请点击“预测变化区域”重新预测。');}};
for(const id of ['search','group','sort','topOnly'])$(id).addEventListener(id==='search'?'input':'change',()=>filterSamples());
for(const id of ['roadLayer','buildingLayer','opacity','layerMode','swipePosition'])$(id).addEventListener('input',draw);
for(const [id,swipe] of [['sideView',false],['swipeView',true]])$(id).onclick=()=>{state.swipe=swipe;$('sideImages').hidden=swipe;$('swipeImages').hidden=!swipe;for(const [b,v] of [['sideView',!swipe],['swipeView',swipe]]){$(b).classList.toggle('active',v);$(b).setAttribute('aria-pressed',String(v));}draw();};
$('benchmarkTab').onclick=()=>setMode('benchmark');$('uploadTab').onclick=()=>setMode('upload');$('exportMask').onclick=exportMask;$('exportCsv').onclick=()=>state.mode==='benchmark'?exportCsv():exportLocalStatistics();
$('beforeFile').onchange=prepareUpload;$('afterFile').onchange=prepareUpload;$('uploadForm').onsubmit=predict;
async function init(){
  const requestedSplit=state.split;const generation=++state.generation;
  const controls=['model','split','search','group','sort','topOnly','exportCsv'];
  controls.forEach(id=>$(id).disabled=true);$('count').textContent='…';$('sampleList').replaceChildren();clearViewer();notice('正在载入完整数据集…');
  try{const response=await fetch('/api/catalog?split='+requestedSplit);if(!response.ok)throw new Error('结果读取失败');const value=await response.json();
    if(requestedSplit!==state.split||generation!==state.generation)return;state.catalog=value;
    $('sampleScope').textContent=`LEVIR-MCI · ${state.catalog.samples.length} 对完整${state.split==='test'?'测试':'验证'}影像`;modelMetadata();filterSamples();
  }catch(error){notice(error.message,true);$('sampleTitle').textContent='数据暂时不可用';}
  finally{controls.forEach(id=>$(id).disabled=false);$('split').disabled=state.mode==='upload';}
}
init();

$('split').onchange=()=>{state.split=$('split').value;state.selected=null;$('search').value='';$('topOnly').checked=false;state.mode='benchmark';$('samplePanel').hidden=false;$('uploadPanel').hidden=true;init();};
