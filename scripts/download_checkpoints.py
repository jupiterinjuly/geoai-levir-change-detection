"""Download the two exact validation-selected checkpoints from the project release."""
from pathlib import Path
import hashlib,json,shutil,urllib.request,ssl
import certifi
R=Path(__file__).resolve().parents[1]
context=ssl.create_default_context(cafile=certifi.where())
base='https://github.com/jupiterinjuly/geoai-levir-change-detection/releases/download/v1.0-course/'
for model,meta in json.loads((R/'data/checkpoints.json').read_text()).items():
 target=R/meta['install_to'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  assert hashlib.sha256(target.read_bytes()).hexdigest()==meta['sha256'],'Existing checkpoint differs; keep your own experiment elsewhere'
 else:
  temp=target.with_suffix('.download')
  with urllib.request.urlopen(base+meta['filename'],context=context,timeout=120) as response,temp.open('wb') as stream:shutil.copyfileobj(response,stream)
  assert temp.stat().st_size==meta['bytes']
  assert hashlib.sha256(temp.read_bytes()).hexdigest()==meta['sha256']
  temp.replace(target)
 print(model,meta['sha256'],'verified')
