"""Get the pinned official dataset, then extract and verify using the original audit."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
(R/'data/raw').mkdir(parents=True,exist_ok=True)
from download_data import main
from prepare_data import extract
main();extract()
print('Pinned archive extracted; original data unchanged. See evidence/ for original audit and approved training exclusions.')
