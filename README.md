# Find Image

Find Image is a local Python tool for identifying what an object or place might be by comparing a target image with a folder of known/reference images.

The current entry point is `find_image.py`. It combines:

- ResNet50 visual similarity
- optional EXIF GPS proximity
- optional manually supplied EXIF tags
- cached image features for faster repeat searches
- JSON, CSV, and montage outputs

Older exploratory versions are preserved in `archive/`.

## Setup

Use Python 3.10 or 3.11 for the smoothest TensorFlow support.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On first search, TensorFlow downloads/caches a ResNet50 ImageNet model at:

```text
~/.landmark_model_cache/resnet50_imagenet.h5
```

## Search Images

```powershell
python find_image.py "path\to\target.jpg" "images" 0.7 0.2
```

With tag matching:

```powershell
python find_image.py "path\to\target.jpg" "images" 0.7 0.2 --tags "church,stone,high street"
```

Useful options:

- `--recursive` searches nested folders.
- `--output-dir outputs` chooses where reports/montages are written.
- `--cache-dir .cache` chooses where image feature cache is stored.
- `--no-cache` recomputes features.
- `--max-results 30` limits montage entries.
- `--max-distance-km 0.2` excludes GPS matches beyond this distance; use `-1` to disable the limit.
- `--verbose` prints per-image decisions.

Each run writes:

- `outputs/find-image-YYYYMMDD-HHMMSS.json`
- `outputs/find-image-YYYYMMDD-HHMMSS.csv`
- `outputs/find-image-YYYYMMDD-HHMMSS.jpg` when matches are found

## Prepare Images With GPS And Tags

`donar.py` copies GPS metadata from a donor image and writes optional comma-separated tags into EXIF `ImageDescription`.

Preview first:

```powershell
python donar.py "path\to\donor_with_gps.jpg" --tags "royal oak,pub,collingham" --dry-run
```

Process by copying originals:

```powershell
python donar.py "path\to\donor_with_gps.jpg" --tags "royal oak,pub,collingham" --copy
```

Process by moving originals from `toprocess/` to `processed/`:

```powershell
python donar.py "path\to\donor_with_gps.jpg" --tags "royal oak,pub,collingham"
```

Custom folders:

```powershell
python donar.py "donor.jpg" --input-folder "toprocess" --output-folder "processed"
```

## Inspect Tags And Distances

```powershell
python readtags.py "images" "path\to\source_with_gps.jpg"
```

This prints stored EXIF tags and distance from a source image when GPS metadata is available.

## Map GPS Clusters

```powershell
python grid.py --image-folder "images" --output "outputs\cluster_map.html"
```

## Project Layout

- `find_image.py` - current image matcher.
- `scoring.py` - dependency-free confidence scoring logic.
- `donar.py` - safer GPS/tag metadata helper with dry-run and copy modes.
- `readtags.py` - metadata inspection helper.
- `grid.py` - GPS cluster map generator.
- `tests/` - built-in unit tests for scoring behavior.
- `archive/` - old `find_image*.py` versions kept for reference.
- `outputs/` - generated reports and montages, ignored by git.
- `.cache/` - cached image embeddings, ignored by git.

Local image folders, generated montages, model weights, ExifTool binaries, and other large/private artifacts are ignored by git by default.

## Tests

```powershell
python -m unittest discover -s tests
```

## Limitations

- Results are similarity suggestions, not definitive identifications.
- GPS scoring only works when both target and candidate images contain usable GPS EXIF metadata.
- Tags are manually managed in EXIF `ImageDescription`.
- HEIC support depends on local Pillow/OS codec support.
- Archived scripts may need extra packages or local files and are not the recommended workflow.
