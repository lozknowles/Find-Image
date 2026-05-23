import argparse
import csv
import json
import os
import pickle
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import piexif
from geopy.distance import geodesic
from PIL import ExifTags, ImageDraw, ImageFont, ImageOps
from PIL import Image as PilImage
from sklearn.metrics.pairwise import cosine_similarity

from scoring import calculate_confidence


VERSION = "5.0.0"
MODEL_NAME = "resnet50_imagenet"
SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MODEL_CACHE_DIR = Path.home() / ".landmark_model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MatchResult:
    rank: int
    file_name: str
    path: str
    confidence: float
    visual_similarity: float
    gps_distance_km: float | None
    gps: tuple[float, float] | None
    matching_tags: list[str]
    all_tags: list[str]


class FeatureCache:
    def __init__(self, cache_path: Path, verbose: bool = False):
        self.cache_path = cache_path
        self.verbose = verbose
        self.data = self._load()
        self.dirty = False

    def _load(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("rb") as cache_file:
                data = pickle.load(cache_file)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            if self.verbose:
                print(f"Could not read feature cache: {exc}")
        return {}

    def get(self, image_path: Path):
        stat = image_path.stat()
        key = str(image_path.resolve())
        cached = self.data.get(key)
        if (
            cached
            and cached.get("mtime") == stat.st_mtime
            and cached.get("size") == stat.st_size
            and cached.get("model") == MODEL_NAME
            and cached.get("version") == VERSION
        ):
            return cached.get("features"), cached.get("tags", []), cached.get("gps")
        return None

    def set(self, image_path: Path, features: np.ndarray, tags: list[str], gps):
        stat = image_path.stat()
        key = str(image_path.resolve())
        self.data[key] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "model": MODEL_NAME,
            "version": VERSION,
            "features": features,
            "tags": tags,
            "gps": gps,
        }
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as cache_file:
            pickle.dump(self.data, cache_file)


def get_landmark_model():
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model, load_model

    model_path = MODEL_CACHE_DIR / f"{MODEL_NAME}.h5"
    if model_path.exists():
        return load_model(model_path)

    print("Downloading ResNet50 model pre-trained on ImageNet...")
    base_model = ResNet50(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def open_image_with_correct_orientation(image_path: Path) -> PilImage.Image:
    img = PilImage.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def read_tags(image_path: Path) -> list[str]:
    try:
        exif_dict = piexif.load(str(image_path))
        tags_value = exif_dict.get("0th", {}).get(piexif.ImageIFD.ImageDescription)
        if not tags_value:
            return []
        if isinstance(tags_value, bytes):
            tags_text = tags_value.decode("utf-8", errors="replace")
        else:
            tags_text = str(tags_value)
        return [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    except Exception:
        return []


def _ratio_to_float(value) -> float:
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _dms_to_decimal(value) -> float:
    return (
        _ratio_to_float(value[0])
        + _ratio_to_float(value[1]) / 60
        + _ratio_to_float(value[2]) / 3600
    )


def extract_gps_info(image_path: Path) -> tuple[float, float] | None:
    try:
        with PilImage.open(image_path) as img:
            exif_data = img._getexif()
        if not exif_data:
            return None
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag != "GPSInfo":
                continue
            gps_info = {
                ExifTags.GPSTAGS.get(key, key): value[key]
                for key in value
            }
            lat = gps_info.get("GPSLatitude")
            lon = gps_info.get("GPSLongitude")
            if not lat or not lon:
                return None
            lat_decimal = _dms_to_decimal(lat)
            lon_decimal = _dms_to_decimal(lon)
            if gps_info.get("GPSLatitudeRef") == "S":
                lat_decimal = -lat_decimal
            if gps_info.get("GPSLongitudeRef") == "W":
                lon_decimal = -lon_decimal
            return (lat_decimal, lon_decimal)
    except Exception:
        return None
    return None


def extract_features(model, image_path: Path) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    img = open_image_with_correct_orientation(image_path)
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x, verbose=0).flatten()


def get_image_record(model, cache: FeatureCache, image_path: Path, use_cache: bool = True):
    if use_cache:
        cached = cache.get(image_path)
        if cached:
            features, tags, gps = cached
            if gps is not None:
                gps = tuple(gps)
            return features, tags, gps

    features = extract_features(model, image_path)
    tags = read_tags(image_path)
    gps = extract_gps_info(image_path)
    if use_cache:
        cache.set(image_path, features, tags, gps)
    return features, tags, gps


def calculate_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    return float(cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0])


def iter_images(folder_path: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def parse_search_tags(tags_text: str | None) -> list[str]:
    if not tags_text:
        return []
    return [tag.strip() for tag in tags_text.split(",") if tag.strip()]


def find_similar_images(
    target_image_path: Path,
    folder_path: Path,
    similarity_threshold: float,
    geo_threshold_km: float,
    search_tags: list[str],
    cache: FeatureCache,
    recursive: bool,
    max_distance_km: float | None,
    use_cache: bool,
    verbose: bool,
) -> list[MatchResult]:
    run_id = uuid.uuid4()
    print(f"Run ID: {run_id}")
    print(f"Target: {target_image_path}")
    print(f"Reference folder: {folder_path}")

    model = get_landmark_model()
    target_features, target_tags, target_gps = get_image_record(model, cache, target_image_path, use_cache)
    if verbose:
        print(f"Target GPS: {target_gps}")
        print(f"Target tags: {target_tags}")

    matches = []
    for image_path in iter_images(folder_path, recursive):
        if image_path.resolve() == target_image_path.resolve():
            continue
        try:
            features, tags, gps = get_image_record(model, cache, image_path, use_cache)
            visual_similarity = calculate_similarity(target_features, features)
            if visual_similarity < similarity_threshold:
                if verbose:
                    print(f"Skip {image_path.name}: similarity {visual_similarity:.4f}")
                continue

            gps_distance_km = geodesic(target_gps, gps).kilometers if target_gps and gps else None
            tag_lookup = {tag.lower() for tag in tags}
            matching_tags = [
                tag for tag in search_tags
                if tag.lower() in tag_lookup
            ]
            confidence = calculate_confidence(
                visual_similarity,
                gps_distance_km,
                geo_threshold_km,
                matching_tags,
                max_distance_km,
            )
            if confidence <= 0:
                if verbose:
                    print(f"Skip {image_path.name}: confidence 0")
                continue

            matches.append(
                MatchResult(
                    rank=0,
                    file_name=image_path.name,
                    path=str(image_path),
                    confidence=confidence,
                    visual_similarity=visual_similarity,
                    gps_distance_km=gps_distance_km,
                    gps=gps,
                    matching_tags=matching_tags,
                    all_tags=tags,
                )
            )
        except Exception as exc:
            print(f"Warning: could not process {image_path}: {exc}", file=sys.stderr)

    matches.sort(key=lambda result: (result.confidence, result.visual_similarity), reverse=True)
    for index, match in enumerate(matches, start=1):
        match.rank = index
    return matches


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json_report(matches: list[MatchResult], output_path: Path, metadata: dict):
    report = {
        "metadata": metadata,
        "matches": [asdict(match) for match in matches],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv_report(matches: list[MatchResult], output_path: Path):
    fieldnames = [
        "rank",
        "file_name",
        "path",
        "confidence",
        "visual_similarity",
        "gps_distance_km",
        "gps",
        "matching_tags",
        "all_tags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            row = asdict(match)
            row["gps"] = json.dumps(row["gps"])
            row["matching_tags"] = ", ".join(row["matching_tags"])
            row["all_tags"] = ", ".join(row["all_tags"])
            writer.writerow(row)


def create_montage(
    matches: list[MatchResult],
    target_image_path: Path,
    output_path: Path,
    metadata: dict,
    max_results: int,
):
    img_width, img_height = 300, 300
    cols = 5
    margin = 12
    header_height = 88
    text_height = 88
    font = ImageFont.load_default()

    selected = matches[:max_results]
    total_images = len(selected) + 1
    rows = (total_images + cols - 1) // cols
    montage_width = cols * (img_width + margin) + margin
    montage_height = header_height + rows * (img_height + text_height + margin) + margin
    montage = PilImage.new("RGB", (montage_width, montage_height), color="white")
    draw = ImageDraw.Draw(montage)

    tag_text = ", ".join(metadata["search_tags"]) if metadata["search_tags"] else "None"
    draw.text((margin, margin), f"Find Image v{VERSION}", fill="black", font=font)
    draw.text(
        (margin, margin + 18),
        f"Target: {Path(metadata['target_image']).name} | Folder: {Path(metadata['folder']).name}",
        fill="black",
        font=font,
    )
    draw.text(
        (margin, margin + 36),
        f"Similarity >= {metadata['similarity_threshold']} | Geo <= {metadata['geo_threshold_km']}km | Tags: {tag_text}",
        fill="black",
        font=font,
    )
    draw.text(
        (margin, margin + 54),
        "Confidence = visual similarity + tag/GPS boosts; see JSON/CSV for exact scores.",
        fill="black",
        font=font,
    )

    def add_tile(image_path: Path, index: int, title: str, confidence, similarity, distance, tags):
        col = index % cols
        row = index // cols
        x = margin + col * (img_width + margin)
        y = header_height + row * (img_height + text_height + margin)
        try:
            img = open_image_with_correct_orientation(image_path)
            img.thumbnail((img_width, img_height))
            montage.paste(img, (x, y))
        except Exception:
            draw.rectangle((x, y, x + img_width, y + img_height), outline="black")
            draw.text((x + 8, y + 8), "Could not load image", fill="black", font=font)

        text_y = y + img_height + 6
        draw.text((x, text_y), title[:44], fill="black", font=font)
        draw.text((x, text_y + 16), f"Conf: {confidence:.4f} | Sim: {similarity:.4f}", fill="black", font=font)
        distance_text = f"{distance:.4f} km" if distance is not None else "No GPS"
        draw.text((x, text_y + 32), f"Distance: {distance_text}", fill="black", font=font)
        draw.text((x, text_y + 48), f"Tags: {', '.join(tags[:3])}", fill="black", font=font)

    add_tile(target_image_path, 0, "Target Image", 1.0, 1.0, None, [])
    for index, match in enumerate(selected, start=1):
        add_tile(
            Path(match.path),
            index,
            f"{match.rank}. {match.file_name}",
            match.confidence,
            match.visual_similarity,
            match.gps_distance_km,
            match.all_tags,
        )

    montage.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find visually and geographically similar images."
    )
    parser.add_argument("target_image", help="Path to the target image")
    parser.add_argument("folder", help="Folder containing images to compare")
    parser.add_argument("similarity_threshold", type=float, help="CNN similarity threshold")
    parser.add_argument("geo_threshold", type=float, help="GPS distance boost threshold in km")
    parser.add_argument("-tag", "--tags", help="Comma-separated tags to search for")
    parser.add_argument("--output-dir", default="outputs", help="Folder for montage and reports")
    parser.add_argument("--cache-dir", default=".cache", help="Folder for cached image features")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature cache reads/writes")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum montage results")
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=0.2,
        help="Set confidence to zero above this GPS distance; use -1 to disable",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-image decisions")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_image_path = Path(args.target_image)
    folder_path = Path(args.folder)
    if not target_image_path.is_file():
        print(f"Error: target image does not exist: {target_image_path}", file=sys.stderr)
        return 2
    if not folder_path.is_dir():
        print(f"Error: folder does not exist: {folder_path}", file=sys.stderr)
        return 2

    search_tags = parse_search_tags(args.tags)
    output_dir = ensure_output_dir(Path(args.output_dir))
    cache = FeatureCache(Path(args.cache_dir) / "feature_cache.pkl", verbose=args.verbose)
    max_distance_km = None if args.max_distance_km < 0 else args.max_distance_km

    started_at = time.strftime("%Y%m%d-%H%M%S")
    metadata = {
        "version": VERSION,
        "target_image": str(target_image_path),
        "folder": str(folder_path),
        "similarity_threshold": args.similarity_threshold,
        "geo_threshold_km": args.geo_threshold,
        "max_distance_km": max_distance_km,
        "search_tags": search_tags,
        "recursive": args.recursive,
    }

    matches = find_similar_images(
        target_image_path=target_image_path,
        folder_path=folder_path,
        similarity_threshold=args.similarity_threshold,
        geo_threshold_km=args.geo_threshold,
        search_tags=search_tags,
        cache=cache,
        recursive=args.recursive,
        max_distance_km=max_distance_km,
        use_cache=not args.no_cache,
        verbose=args.verbose,
    )
    cache.save()

    prefix = f"find-image-{started_at}"
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    montage_path = output_dir / f"{prefix}.jpg"

    write_json_report(matches, json_path, metadata)
    write_csv_report(matches, csv_path)
    if matches:
        create_montage(matches, target_image_path, montage_path, metadata, args.max_results)

    print(f"Matches: {len(matches)}")
    if matches:
        for match in matches[:10]:
            distance = f"{match.gps_distance_km:.4f} km" if match.gps_distance_km is not None else "No GPS"
            print(
                f"{match.rank:>2}. {match.file_name} "
                f"conf={match.confidence:.4f} sim={match.visual_similarity:.4f} dist={distance}"
            )
        print(f"Montage: {montage_path}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
