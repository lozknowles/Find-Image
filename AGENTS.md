# AGENTS.md

Guidance for AI coding agents working in this project.

## Project Purpose

Find Image is a local Python prototype for identifying unknown objects/places by comparing a target image with a reference folder using visual embeddings, EXIF GPS proximity, and optional manual tags.

Use root-level `find_image.py` as the active implementation. Older `find_image*.py` versions live in `archive/` and should be treated as reference material.

## Working Practices

- Keep generated files out of git: `.cache/`, `outputs/`, model weights, local images, videos, and ExifTool binaries are intentionally ignored.
- Do not publish personal/reference image folders unless the user explicitly asks.
- Be careful with EXIF metadata. `donar.py` can mutate files and move/copy images.
- Prefer adding tests or small helpers around `find_image.py` instead of editing archived scripts.
- Keep command examples Windows-friendly unless the user asks otherwise.

## Main Commands

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the matcher:

```powershell
python find_image.py "path\to\target.jpg" "images" 0.7 0.2 --tags "tag one,tag two"
```

Disable the hard GPS exclusion:

```powershell
python find_image.py "path\to\target.jpg" "images" 0.7 0.2 --max-distance-km -1
```

Preview EXIF writes:

```powershell
python donar.py "path\to\donor_with_gps.jpg" --tags "tag one,tag two" --dry-run
```

Map GPS clusters:

```powershell
python grid.py --image-folder "images" --output "outputs\cluster_map.html"
```

## Implementation Notes

- `find_image.py` lazily imports TensorFlow/Keras only when a search actually runs.
- `scoring.py` contains dependency-free confidence scoring and is covered by `python -m unittest discover -s tests`.
- Image features are cached in `.cache/feature_cache.pkl`, keyed by resolved path, modified time, file size, model name, and script version.
- Each search writes JSON and CSV reports, plus a montage when matches are found.
- Confidence starts from cosine visual similarity and adds boosts for matching tags and GPS proximity.
- The identical-GPS boost is evaluated before the near-GPS boost.
- `--max-distance-km` defaults to `0.2`, preserving the old behavior while making it configurable.

## GitHub Notes

When committing this project, include code and docs. Avoid adding ignored media/model/binary artifacts unless the user explicitly wants a data-heavy repository.
