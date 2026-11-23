from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json,csv,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import Rectangle,Patch
from common import ROOT,PALETTE,load_triplet
O=ROOT/'local_runs/regenerated_figures';O.mkdir(parents=True,exist_ok=True)
font='/System/Library/Fonts/STHeiti Light.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,'axes.unicode_minus':False,'axes.edgecolor':'black','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','savefig.dpi':220})
def save(fig,n):
 fig.savefig(O/(n+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(O/(n+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
C=['#0072B2','#D55E00']
M=['baseline','attention'];L=['基础U-Net','自注意力U-Net'];S={m:json.load(open(ROOT/f'evidence/test-analysis-{m}.json')) for m in M}
f,ax=plt.subplots(figsize=(12,5));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,5))
def box(x,y,t,w=2.1):
 ax.add_patch(Rectangle((x,y),w,.55,facecolor='white',edgecolor='black',lw=.9));ax.text(x+w/2,y+.275,t,ha='center',va='center',fontsize=9)
def arrow(a,b,dash=False):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':.8,'color':'black','linestyle':'--' if dash else '-'})
box(.1,4.1,'前期RGB＋后期RGB\n6×256×256',2.7);arrow((1.45,4.1),(1.45,3.75))
for i,(r,c) in enumerate([(256,16),(128,32),(64,64),(32,128)]):
 y=3.2-i*.8;box(.4,y,f'{r}×{r}×{c}');box(9.3,y,f'{r}×{r}×{c}');arrow((2.5,y+.27),(9.3,y+.27),True)
 if i<3:arrow((1.45,y),(1.45,y-.25));arrow((10.35,y-.25),(10.35,y))
box(3.4,.1,'瓶颈 16×16×256',2.4);box(6.15,.1,'基础：直接传递\n增强：4头自注意力＋FFN',3.3)
arrow((1.45,.8),(1.45,.38));arrow((1.45,.38),(3.4,.38));arrow((5.8,.38),(6.15,.38));arrow((9.45,.38),(10.35,.38));arrow((10.35,.38),(10.35,.8))
box(9.1,4.1,'背景／道路变化／建筑变化',2.7);arrow((10.35,3.75),(10.35,4.1));ax.text(5.9,3.6,'同尺度跳跃连接（虚线）',ha='center',fontsize=10)
f.tight_layout();save(f,'图1_模型结构')
f,axes=plt.subplots(1,2,figsize=(11,3.5))
for m,label,col,ls,mk in zip(M,L,C,['-','--'],['o','s']):
 h=json.load(open(ROOT/f'runs/formal_{m}_seed42/summary.json'))['history'];e=[x['epoch'] for x in h]
 axes[0].plot(e,[x['train_loss'] for x in h],color=col,ls=ls,label=label+'训练')
 axes[0].plot(e,[x['validation']['loss'] for x in h],color=col,ls=':' if m=='baseline' else '-.',label=label+'验证')
 axes[1].plot(e,[100*x['validation']['foreground_miou'] for x in h],color=col,ls=ls,marker=mk,markevery=3,markersize=3,label=label)
for a,t,y in zip(axes,['训练与验证损失','验证集前景平均IoU'],['损失','前景平均IoU/%']):
 a.set(title=t,xlabel='训练轮次',ylabel=y,xlim=(1,30));a.spines[['top','right']].set_visible(False);a.legend(fontsize=8,framealpha=.95);a.grid(axis='y',color='.88',lw=.5)
axes[1].set_ylim(0,100);f.tight_layout();save(f,'图2_训练与验证曲线')
f,axes=plt.subplots(1,2,figsize=(10,3.5))
for a,key,title in zip(axes,['iou','f1'],['IoU','F1']):
 x=np.arange(2)
 for j,m in enumerate(M):
  vs=np.array(S[m][key][1:])*100;b=a.bar(x+(j-.5)*.34,vs,.34,label=L[j],color=C[j],alpha=.85,edgecolor='black',hatch='' if j==0 else '//',linewidth=.8)
  for z,v in zip(b,vs):a.text(z.get_x()+z.get_width()/2,v+1,f'{v:.2f}',ha='center',fontsize=8)
 a.set(xticks=x,xticklabels=['道路变化','建筑变化'],ylabel=title+'/%',ylim=(0,100));a.spines[['top','right']].set_visible(False);a.legend(fontsize=8)
f.tight_layout();save(f,'图3_测试指标')
samples=json.load(open(ROOT/'evidence/final-test-case-samples.json'))['samples']
for subset,n in [(samples[:3],'图4a_典型案例'),(samples[3:],'图4b_失败案例')]:
 f,axes=plt.subplots(3,5,figsize=(13,7.3))
 for i,v in enumerate(subset):
  a,b,y=load_triplet('test',v['filename']);ims=[a,b,PALETTE[y],*[np.array(Image.open(ROOT/S[m]['folder']/'predictions_rgb'/v['filename'])) for m in M]]
  for ax,im in zip(axes[i],ims):ax.imshow(im);ax.axis('off')
  axes[i,0].text(0,-.06,v['filename'].replace('.png',''),transform=axes[i,0].transAxes,fontsize=8)
 for ax,t in zip(axes[0],['前期影像','后期影像','参考标签',*L]):ax.set_title(t,fontsize=10)
 f.tight_layout();save(f,n)
err=json.load(open(ROOT/'evidence/report-road-error-cases.json'))['samples'];f,axes=plt.subplots(3,5,figsize=(13,7.3))
colors=np.array([[255,255,255],[160,160,160],[213,94,0],[0,114,178]],dtype=np.uint8)
for i,n in enumerate(err):
 a,b,y=load_triplet('test',n);truth=y==1;gt=np.ones_like(a)*255;gt[truth]=[0,0,0];errors=[]
 for m in M:
  pred=np.array(Image.open(ROOT/S[m]['folder']/'predictions_class_id'/n))==1;idx=np.zeros_like(y);idx[truth&pred]=1;idx[~truth&pred]=2;idx[truth&~pred]=3;errors.append(colors[idx])
 for ax,im in zip(axes[i],[a,b,gt,*errors]):ax.imshow(im);ax.axis('off')
 axes[i,0].text(0,-.06,n.replace('.png',''),transform=axes[i,0].transAxes,fontsize=8)
for ax,t in zip(axes[0],['前期影像','后期影像','道路参考标签',L[0]+'错误',L[1]+'错误']):ax.set_title(t,fontsize=9)
f.legend(handles=[Patch(facecolor=c/255,label=t) for c,t in zip(colors[1:],['正确命中','误报','漏检'])],loc='lower center',ncol=3,frameon=False,fontsize=9);f.tight_layout(rect=(0,.04,1,1));save(f,'图5_道路错误分析')
rows={m:list(csv.DictReader((ROOT/S[m]['folder']/'per-image.csv').open())) for m in M};f,axes=plt.subplots(1,2,figsize=(10,4))
for ax,(cls,t) in zip(axes,[('road','道路变化'),('building','建筑变化')]):
 for m,col,mk,l in zip(M,C,['.','x'],L):
  ax.scatter([float(v[f'true_{cls}_fraction'])*100 for v in rows[m]],[float(v[f'pred_{cls}_fraction'])*100 for v in rows[m]],s=13,alpha=.50,color=col,marker=mk,label=l,linewidths=.65)
 ax.plot([0,100],[0,100],'k--',lw=.8);ax.set(xlim=(0,100),ylim=(0,100),xlabel='参考占比/%',ylabel='预测占比/%',title=t);ax.set_aspect('equal');ax.legend(fontsize=8,markerscale=1.8,framealpha=.95)
f.tight_layout();save(f,'图6_面积估计')
f,ax=plt.subplots(figsize=(7,3.7))
for m,col,ls,l in zip(M,C,['-','--'],L):
 rr=sorted(rows[m],key=lambda x:(-float(x['pred_change_pixels'])/float(x['pixels']),x['filename']));vv=np.array([int(x['true_change_pixels']) for x in rr]);yy=np.r_[0,np.cumsum(vv)]/vv.sum()*100;xx=np.arange(len(rr)+1)/len(rr)*100;ax.plot(xx,yy,color=col,ls=ls,label=l+f'（前20%：{yy[386]:.2f}%）',lw=1.8);ax.plot(xx[386],yy[386],marker='o' if m=='baseline' else 's',color=col,ms=4)
ax.plot([0,100],[0,100],color='.6',ls=':',label='随机查看的覆盖率期望');ax.axvline(20,color='.5',ls=':',lw=.7);ax.set(xlim=(0,100),ylim=(0,100),xlabel='优先查看影像比例/%',ylabel='真实变化像素覆盖率/%');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,markerscale=1.8,framealpha=.95);f.tight_layout();save(f,'图7_排序覆盖率')
print('Academic figures regenerated from saved results; no experiments rerun')

manifest=list(csv.DictReader((ROOT/'evidence/proposed-clean-manifest.csv').open()))
f,axes=plt.subplots(1,4,figsize=(12,3),gridspec_kw={'width_ratios':[1,1,1,1.3]})
a,b,y=load_triplet('train','train_000040.png')
for ax,im,t in zip(axes[:3],[a,b,PALETTE[y]],['训练样本前期','训练样本后期','参考变化标签']):
 ax.imshow(im);ax.set_title(t);ax.axis('off')
ax=axes[3];x=np.arange(3);bottom=np.zeros(3)
for c,col,l in zip(range(3),['#D9D9D9','#E69F00','#CC6677'],['背景','道路变化','建筑变化']):
 v=[]
 for split in ['train','val','test']:
  rr=[r for r in manifest if r['split']==split];v.append(sum(int(r[f'pixels_{c}']) for r in rr)/(len(rr)*65536)*100)
 ax.bar(x,v,bottom=bottom,color=col,label=l,width=.6,edgecolor='white');bottom+=v
ax.set(xticks=x,xticklabels=['训练','验证','测试'],ylabel='像素比例/%',ylim=(0,100));ax.legend(fontsize=7,loc='lower left');ax.spines[['top','right']].set_visible(False)
f.tight_layout();save(f,'图0_数据概览')
