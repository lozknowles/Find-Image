import os
import argparse
import folium
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from sklearn.cluster import DBSCAN
import numpy as np

def get_gps_data(image_path):
    try:
        with Image.open(image_path) as img:
            raw_exif = img._getexif()
        if not raw_exif:
            return None
        exif = {TAGS[k]: v for k, v in raw_exif.items() if k in TAGS}
        if 'GPSInfo' in exif:
            gps_info = {GPSTAGS[k]: v for k, v in exif['GPSInfo'].items()}
            lat = gps_info['GPSLatitude']
            lon = gps_info['GPSLongitude']
            lat = lat[0] + lat[1]/60 + lat[2]/3600
            lon = lon[0] + lon[1]/60 + lon[2]/3600
            if gps_info['GPSLatitudeRef'] == 'S':
                lat = -lat
            if gps_info['GPSLongitudeRef'] == 'W':
                lon = -lon
            return lat, lon
    except Exception:
        return None
    return None

def build_map(image_folder, output_path, eps, min_samples):
    gps_data = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp')):
            gps = get_gps_data(os.path.join(image_folder, filename))
            if gps:
                gps_data.append(gps)

    if not gps_data:
        print("No GPS data found in the images.")
        return

    coords = np.array(gps_data)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    center_lat, center_lon = coords.mean(axis=0)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    for label, color in zip(set(labels), colors):
        if label == -1:
            continue
        cluster_points = coords[labels == label]
        cluster_center = cluster_points.mean(axis=0)
        cluster_size = len(cluster_points)
        folium.CircleMarker(
            location=cluster_center,
            radius=min(20, max(5, cluster_size)),
            popup=f'Cluster {label}: {cluster_size} images',
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    bounds = m.get_bounds()
    lat_min, lon_min = bounds['_southWest']['lat'], bounds['_southWest']['lng']
    lat_max, lon_max = bounds['_northEast']['lat'], bounds['_northEast']['lng']
    
    lat_step = (lat_max - lat_min) / 5
    lon_step = (lon_max - lon_min) / 5
    
    for i in range(5):
        for j in range(5):
            zone_num = i * 5 + j + 1
            lat = lat_min + (i + 0.5) * lat_step
            lon = lon_min + (j + 0.5) * lon_step
            folium.Rectangle(
                bounds=[[lat - lat_step/2, lon - lon_step/2], [lat + lat_step/2, lon + lon_step/2]],
                color='black',
                weight=1,
                fill=False,
            ).add_to(m)
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 14pt; color: rgba(0, 0, 0, 0.5);">{zone_num}</div>'
                )
            ).add_to(m)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    print(f"Map saved as {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a map of image GPS clusters.")
    parser.add_argument("--image-folder", default="images", help="Folder containing images to map")
    parser.add_argument("--output", default=os.path.join("outputs", "cluster_map.html"), help="HTML map output path")
    parser.add_argument("--eps", type=float, default=0.1, help="DBSCAN epsilon value")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN minimum samples")
    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"Error: image folder does not exist: {args.image_folder}")
        return 2
    build_map(args.image_folder, args.output, args.eps, args.min_samples)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
