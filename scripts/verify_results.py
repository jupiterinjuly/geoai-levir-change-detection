"""Verify archived metrics independently using only the Python standard library."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
protocol=read('configs/final-test-approved.json')
for key in ['manifest','normalization']:
 assert hashlib.sha256((ROOT/protocol[key]).read_bytes()).hexdigest()==protocol[key+'_sha256']
for model in ['baseline','attention']:
 s=read('evidence/test-analysis-'+model+'.json')
 rows=list(csv.DictReader((ROOT/s['folder']/'per-image.csv').open()))
 assert len(rows)==s['samples']==1929
 assert [r['filename'] for r in rows]==protocol['test_filenames']
 cm=s['confusion_matrix'];assert sum(map(sum,cm))==1929*65536
 for c,cls in [(1,'road'),(2,'building')]:
  tp=sum(int(r[cls+'_tp']) for r in rows);fp=sum(int(r[cls+'_fp']) for r in rows);fn=sum(int(r[cls+'_fn']) for r in rows)
  assert tp==cm[c][c] and fp==sum(x[c] for x in cm)-tp and fn==sum(cm[c])-tp
  near(s['iou'][c],tp/(tp+fp+fn));near(s['precision'][c],tp/(tp+fp));near(s['recall'][c],tp/(tp+fn));near(s['f1'][c],2*tp/(2*tp+fp+fn))
  mae=sum(100*abs(int(r['pred_'+cls+'_pixels'])-int(r['true_'+cls+'_pixels']))/int(r['pixels']) for r in rows)/len(rows)
  near(s[cls+'_area_mae_pp'],mae)
 near(s['foreground_miou'],sum(s['iou'][1:])/2)
 ranked=sorted(rows,key=lambda r:(-int(r['pred_change_pixels'])/int(r['pixels']),r['filename']))
 selected=ranked[:math.ceil(len(rows)*.2)]
 near(s['ranking_top20pct']['true_change_pixel_coverage'],sum(int(r['true_change_pixels']) for r in selected)/sum(int(r['true_change_pixels']) for r in rows))
 empty=[r for r in rows if int(r['true_change_pixels'])==0];assert len(empty)==953
 near(s['empty_false_alarm_image_rate'],sum(int(r['pred_change_pixels'])>0 for r in empty)/len(empty))
 for group,stat in s['size_groups'].items():
  rs=[r for r in rows if r['size_group']==group];assert len(rs)==stat['images']
  scores=[]
  for cls in ['road','building']:
   tp,fp,fn=[sum(int(r[cls+'_'+k]) for r in rs) for k in ['tp','fp','fn']]
   if tp+fp+fn:scores.append(tp/(tp+fp+fn))
  if scores:near(stat['metrics']['foreground_miou'],sum(scores)/len(scores))
 for r in rows:
  assert int(r['pixels'])==65536
  assert int(r['true_change_pixels'])==int(r['true_road_pixels'])+int(r['true_building_pixels'])
  assert int(r['pred_change_pixels'])==int(r['pred_road_pixels'])+int(r['pred_building_pixels'])
 print(model+': PASS - 1,929 rows, confusion, IoU/P/R/F1, area, groups, ranking and empty-image metrics')
manifest=ROOT/'data/public_files_sha256.json'
if manifest.exists():
 for rel,h in json.loads(manifest.read_text()).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
 print('Public archive SHA256 manifest: PASS')
print('This verifies archived internal consistency, not an independent retraining or label-semantic validation.')
