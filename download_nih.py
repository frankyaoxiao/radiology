"""Download NIH ChestXray14 images from HuggingFace."""
from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile, sys

DST = Path('/data/artifacts/frank/misc/nih_chest14')
IMG_DIR = DST / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

CACHE = '/data/artifacts/frank/misc/nih_cache'

for i in range(1, 13):
    name = f'images_{i:03d}.zip'
    print(f'downloading {name}...', flush=True)
    p = hf_hub_download(repo_id='alkzar90/NIH-Chest-X-ray-dataset',
                        filename=f'data/images/{name}',
                        repo_type='dataset', cache_dir=CACHE)
    print(f'  cached at {p}', flush=True)
    print(f'  extracting...', flush=True)
    with zipfile.ZipFile(p) as zf:
        members = [m for m in zf.namelist() if m.endswith('.png')]
        # NIH images are at root of zip: 00000001_000.png etc.
        for m in members:
            # Strip any prefix; flatten to images/
            target = IMG_DIR / Path(m).name
            if not target.exists():
                with zf.open(m) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
    print(f'  extracted', flush=True)

# Count
total = len(list(IMG_DIR.glob('*.png')))
print(f'\nTOTAL PNG files: {total}', flush=True)
