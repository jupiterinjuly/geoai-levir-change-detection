from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from common import ROOT,PALETTE,load_triplet
O=ROOT/'local_runs/regenerated_figures';O.mkdir(parents=True,exist_ok=True)
font='/System/Library/Fonts/STHeiti Light.ttc';font_manager.fontManager.addfont(font)
C=['#526E8B','#6E9D92'];INK='#29333C';GRID='#E7EBEE';TP='#A6B8AD';FP='#B97D7C';FN='#6686A1';BG='#F3F5F6'
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,'axes.titlesize':11,'axes.titleweight':'normal','axes.labelsize':10,'axes.labelpad':5,'axes.unicode_minus':False,'axes.edgecolor':'#9AA4AC','axes.linewidth':.65,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'xtick.major.size':3,'ytick.major.size':3,'legend.frameon':False,'legend.fontsize':9,'savefig.dpi':300,'svg.fonttype':'none'})
M=['baseline','attention'];L=['基础 U-Net','自注意力 U-Net'];S={m:json.load(open(ROOT/f'evidence/test-analysis-{m}.json')) for m in M}
def save(fig,n):
 fig.savefig(O/(n+'.png'),bbox_inches='tight',pad_inches=.06,facecolor='white');fig.savefig(O/(n+'.svg'),bbox_inches='tight',pad_inches=.06,facecolor='white');plt.close(fig)
def clean(ax,grid=True):
 ax.spines[['top','right']].set_visible(False)
 if grid:ax.grid(axis='y',color=GRID,lw=.6,zorder=0);ax.set_axisbelow(True)
 return ax
def legend(models=True):return [Line2D([0],[0],color=c,lw=1.7,marker=mk,ms=4,label=l) for c,mk,l in zip(C,['o','D'],L)]
# Structure: quiet fills distinguish encoder, bottleneck and decoder.
f,ax=plt.subplots(figsize=(9.1,3.65));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,5))
def box(x,y,t,w=2.1,fill='#EEF2F5',edge='#A2B0BA'):
 ax.add_patch(Rectangle((x,y),w,.55,facecolor=fill,edgecolor=edge,lw=.7));ax.text(x+w/2,y+.275,t,ha='center',va='center',fontsize=8.3)
def arrow(a,b,dash=False):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':.7,'color':'#82909B','linestyle':'--' if dash else '-'})
box(.1,4.1,'前期 RGB＋后期 RGB\n6×256×256',2.7);arrow((1.45,4.1),(1.45,3.75))
for i,(r,c) in enumerate([(256,16),(128,32),(64,64),(32,128)]):
 y=3.2-i*.8;box(.4,y,f'{r}×{r}×{c}');box(9.3,y,f'{r}×{r}×{c}',fill='#EEF4F1',edge='#9FB5AA');arrow((2.5,y+.27),(9.3,y+.27),True)
 if i<3:arrow((1.45,y),(1.45,y-.25));arrow((10.35,y-.25),(10.35,y))
box(3.4,.1,'瓶颈 16×16×256',2.4);box(6.15,.1,'基础：直接传递\n增强：4 头自注意力＋FFN',3.3,fill='#E5EFEB',edge='#8FAEA1')
for a,b in [((1.45,.8),(1.45,.38)),((1.45,.38),(3.4,.38)),((5.8,.38),(6.15,.38)),((9.45,.38),(10.35,.38)),((10.35,.38),(10.35,.8))]:arrow(a,b)
box(9.1,4.1,'背景／道路变化／建筑变化',2.7,fill='#EEF4F1');arrow((10.35,3.75),(10.35,4.1));ax.text(5.9,3.6,'同尺度跳跃连接',ha='center',fontsize=9,color='#667782');f.tight_layout(pad=.3);save(f,'图1_模型结构')
# Three small multiples replace a four-line loss plot and its busy legend.
f,axes=plt.subplots(1,3,figsize=(8.9,2.65));hist={m:json.load(open(ROOT/f'runs/formal_{m}_seed42/summary.json'))['history'] for m in M}
for j,(field,title,ylabel) in enumerate([('train_loss','（a）训练损失','损失'),('val_loss','（b）验证损失','损失'),('miou','（c）验证集前景平均 IoU','IoU（%）')]):
 ax=axes[j]
 for m,c,ls,mk in zip(M,C,['-','--'],['o','D']):
  h=hist[m];v=[x['train_loss'] if field=='train_loss' else x['validation']['loss'] if field=='val_loss' else 100*x['validation']['foreground_miou'] for x in h]
  ax.plot([x['epoch'] for x in h],v,color=c,ls=ls,lw=1.65,marker=mk,markevery=6,ms=3.5,mfc='white' if m=='attention' else c,mew=.8)
 ax.set(title=title,xlabel='训练轮次',ylabel=ylabel,xlim=(1,30),xticks=[1,10,20,30]);clean(ax)
 if field=='miou':ax.set(ylim=(0,100),yticks=[0,25,50,75,100])
f.legend(handles=legend(),loc='lower center',bbox_to_anchor=(.52,-.04),ncol=2,columnspacing=2.8);f.tight_layout(rect=(0,.1,1,1),w_pad=1.8);save(f,'图2_训练与验证曲线')
# No hatching or heavy bar outlines; full 0-100 scale retained.
f,axes=plt.subplots(1,2,figsize=(8.6,2.8))
for ax,key,title in zip(axes,['iou','f1'],['（a）交并比 IoU','（b）F1 分数']):
 x=np.arange(2)
 for j,m in enumerate(M):
  vals=np.array(S[m][key][1:])*100;b=ax.bar(x+(j-.5)*.25,vals,.23,color=C[j],edgecolor='none',zorder=3)
  for p,v in zip(b,vals):ax.text(p.get_x()+p.get_width()/2,v+2,f'{v:.2f}',ha='center',va='bottom',fontsize=9,color=C[j],fontweight='medium')
 ax.set(title=title,xticks=x,xticklabels=['道路变化','建筑变化'],ylabel='指标（%）',ylim=(0,100),yticks=[0,25,50,75,100],xlim=(-.55,1.55));clean(ax)
f.legend(handles=[Patch(facecolor=c,edgecolor='none',label=l) for c,l in zip(C,L)],loc='lower center',ncol=2,bbox_to_anchor=(.52,-.035),columnspacing=2.8);f.tight_layout(rect=(0,.1,1,1),w_pad=2);save(f,'图3_测试指标')
# Comparison plates retain original annotation colors and the actual saved predictions.
samples=json.load(open(ROOT/'evidence/final-test-case-samples.json'))['samples']
for subset,n in [(samples[:3],'图4a_典型案例'),(samples[3:],'图4b_失败案例')]:
 f,axes=plt.subplots(3,5,figsize=(10.4,5.85))
 for i,v in enumerate(subset):
  a,b,y=load_triplet('test',v['filename']);ims=[a,b,PALETTE[y],*[np.array(Image.open(ROOT/S[m]['folder']/'predictions_rgb'/v['filename'])) for m in M]]
  for ax,im in zip(axes[i],ims):ax.imshow(im);ax.set(xticks=[],yticks=[]);[sp.set(color='#D7DFE3',linewidth=.5) for sp in ax.spines.values()]
  axes[i,0].text(0,-.08,v['filename'].replace('.png',''),transform=axes[i,0].transAxes,fontsize=7.8,color='#5C6872')
 for ax,t in zip(axes[0],['前期影像','后期影像','参考标签',*L]):ax.set_title(t,fontsize=8.8,pad=5)
 f.subplots_adjust(left=.015,right=.99,top=.94,bottom=.025,wspace=.08,hspace=.17);save(f,n)
# Error map: explicit panel boundaries and empty-panel annotations.
err=json.load(open(ROOT/'evidence/report-road-error-cases.json'))['samples'];f,axes=plt.subplots(3,5,figsize=(10.4,5.85))
from matplotlib.colors import to_rgb
colors=np.round(np.array([to_rgb(c) for c in [BG,TP,FP,FN]])*255).astype(np.uint8)
for i,n in enumerate(err):
 a,b,y=load_triplet('test',n);truth=y==1;gt=np.zeros_like(a);gt[:]=colors[0];gt[truth]=np.round(np.array(to_rgb(C[0]))*255).astype(np.uint8);errors=[]
 for m in M:
  pred=np.array(Image.open(ROOT/S[m]['folder']/'predictions_class_id'/n))==1;idx=np.zeros_like(y);idx[truth&pred]=1;idx[~truth&pred]=2;idx[truth&~pred]=3;errors.append(colors[idx])
 for k,(ax,im) in enumerate(zip(axes[i],[a,b,gt,*errors])):
  ax.imshow(im);ax.set(xticks=[],yticks=[]);[sp.set(color='#D7DFE3',linewidth=.55) for sp in ax.spines.values()]
  if k==2 and not truth.any():ax.text(.5,.5,'无道路变化标签',ha='center',va='center',transform=ax.transAxes,fontsize=8,color='#87919A')
 axes[i,0].text(0,-.08,n.replace('.png',''),transform=axes[i,0].transAxes,fontsize=7.8,color='#5C6872')
for ax,t in zip(axes[0],['前期影像','后期影像','道路参考标签','基础模型错误','自注意力模型错误']):ax.set_title(t,fontsize=8.8,pad=5)
f.legend(handles=[Patch(facecolor=c,edgecolor='none',label=t) for c,t in zip([TP,FP,FN],['正确命中','误报','漏检'])],loc='lower center',ncol=3,bbox_to_anchor=(.51,-.005),columnspacing=2.4,handlelength=1.3)
f.subplots_adjust(left=.015,right=.99,top=.94,bottom=.07,wspace=.08,hspace=.17);save(f,'图5_道路错误分析')
# Scatter: muted color, small transparent markers and equal aspect, no point jitter.
rows={m:list(csv.DictReader((ROOT/S[m]['folder']/'per-image.csv').open())) for m in M};f,axes=plt.subplots(1,2,figsize=(8.4,3.35))
for ax,(cls,t) in zip(axes,[('road','（a）道路变化'),('building','（b）建筑变化')]):
 ax.plot([0,100],[0,100],color='#9CA6AE',ls=(0,(4,3)),lw=.85,zorder=1)
 for m,c,mk in zip(M,C,['o','D']):
  ax.scatter([float(v[f'true_{cls}_fraction'])*100 for v in rows[m]],[float(v[f'pred_{cls}_fraction'])*100 for v in rows[m]],s=10,alpha=.32,c=c,marker=mk,linewidths=0,zorder=2)
 ax.set(xlim=(0,100),ylim=(0,100),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],xlabel='参考占比（%）',ylabel='预测占比（%）',title=t);ax.set_aspect('equal');clean(ax,False)
f.legend(handles=[Line2D([0],[0],color=c,ls='none',marker=mk,ms=4,label=l) for c,mk,l in zip(C,['o','D'],L)],loc='lower center',ncol=2,bbox_to_anchor=(.52,-.025),columnspacing=2.8);f.tight_layout(rect=(0,.07,1,1),w_pad=3.0);save(f,'图6_面积估计')
# Coverage curves overlap in reality: preserve the overlap and report exact values in legend.
f,ax=plt.subplots(figsize=(6.4,2.9))
for m,c,ls,mk,l in zip(M,C,['-','--'],['o','D'],L):
 rr=sorted(rows[m],key=lambda x:(-float(x['pred_change_pixels'])/float(x['pixels']),x['filename']));vv=np.array([int(x['true_change_pixels']) for x in rr]);yy=np.r_[0,np.cumsum(vv)]/vv.sum()*100;xx=np.arange(len(rr)+1)/len(rr)*100
 ax.plot(xx,yy,color=c,ls=ls,label=l+f'  {yy[386]:.2f}%',lw=1.8);ax.plot(xx[386],yy[386],marker=mk,color=c,ms=4,mfc='white' if m=='attention' else c)
ax.plot([0,100],[0,100],color='#BBC3C8',ls=(0,(2,3)),label='随机查看期望',lw=1.1);ax.axvline(20,color='#D1D8DC',ls=(0,(3,3)),lw=.8)
ax.set(xlim=(0,100),ylim=(0,100),xticks=[0,20,40,60,80,100],yticks=[0,25,50,75,100],xlabel='优先查看影像比例（%）',ylabel='真实变化像素覆盖率（%）');clean(ax)
ax.legend(loc='lower right',fontsize=8.5,title='前 20% 影像覆盖率',title_fontsize=8.5,labelspacing=.65,handlelength=2.8);f.tight_layout(pad=.6);save(f,'图7_排序覆盖率')
# Distribution figure uses muted semantic colors; raw labels remain unchanged.
manifest=list(csv.DictReader((ROOT/'evidence/proposed-clean-manifest.csv').open()));f,axes=plt.subplots(1,4,figsize=(10,2.45),gridspec_kw={'width_ratios':[1,1,1,1.3]})
a,b,y=load_triplet('train','train_000040.png')
for ax,im,t in zip(axes[:3],[a,b,PALETTE[y]],['训练样本前期','训练样本后期','参考变化标签']):ax.imshow(im);ax.set_title(t,fontsize=9);ax.axis('off')
ax=axes[3];x=np.arange(3);bottom=np.zeros(3)
for c,col,l in zip(range(3),['#DCE2E5','#BBA777','#809E91'],['背景','道路变化','建筑变化']):
 v=[]
 for split in ['train','val','test']:
  rr=[r for r in manifest if r['split']==split];v.append(sum(int(r[f'pixels_{c}']) for r in rr)/(len(rr)*65536)*100)
 ax.bar(x,v,bottom=bottom,color=col,label=l,width=.55,edgecolor='white',linewidth=.4);bottom+=v
ax.set(xticks=x,xticklabels=['训练','验证','测试'],ylabel='像素比例（%）',ylim=(0,100));clean(ax,False);ax.legend(fontsize=7.8,loc='lower left',labelspacing=.45,handlelength=1)
f.tight_layout(pad=.5,w_pad=1.0);save(f,'图0_数据概览')
print('Muted academic figures created from archived results; no experiment modified')
