"""Get the pinned official dataset, then extract and verify using the original audit."""
import sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
(R/'data/raw').mkdir(parents=True,exist_ok=True)
import download_data
metadata=R/'local_runs/data_download';metadata.mkdir(parents=True,exist_ok=True)
for name in ['huggingface-dataset.json','huggingface-files.json']:
 shutil.copy2(R/'evidence'/name,metadata/name)
download_data.META=metadata
from prepare_data import extract
download_data.main();extract()
print('Pinned archive extracted; original data unchanged. See evidence/ for original audit and approved training exclusions.')
