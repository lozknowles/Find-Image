import os
import argparse
from PIL import Image
import piexif
from math import radians, sin, cos, sqrt, atan2

def read_image_tags(image_path):
    try:
        exif_dict = piexif.load(image_path)
        if piexif.ImageIFD.ImageDescription in exif_dict["0th"]:
            tags_str = exif_dict["0th"][piexif.ImageIFD.ImageDescription].decode('utf-8')
            return [tag.strip() for tag in tags_str.split(',')]
        else:
            return []
    except Exception as e:
        print(f"Error reading tags from {image_path}: {str(e)}")
        return []

def get_geo_data(exif_dict):
    if "GPS" in exif_dict:
        gps = exif_dict["GPS"]
        if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
            lat = gps[piexif.GPSIFD.GPSLatitude]
            lon = gps[piexif.GPSIFD.GPSLongitude]
            lat_ref = gps[piexif.GPSIFD.GPSLatitudeRef].decode('utf-8')
            lon_ref = gps[piexif.GPSIFD.GPSLongitudeRef].decode('utf-8')
            
            lat = (lat[0][0] / lat[0][1] + lat[1][0] / lat[1][1] / 60 + lat[2][0] / lat[2][1] / 3600) * (-1 if lat_ref == "S" else 1)
            lon = (lon[0][0] / lon[0][1] + lon[1][0] / lon[1][1] / 60 + lon[2][0] / lon[2][1] / 3600) * (-1 if lon_ref == "W" else 1)
            
            return lat, lon
    return None

def calculate_distance(coord1, coord2):
    R = 6371  # Earth's radius in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def process_folder(folder_path, source_geo):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            tags = read_image_tags(file_path)
            
            print(f"{filename}:")
            print(f"  Tags: {', '.join(tags) if tags else 'No tags found'}")
            
            try:
                if filename.lower().endswith(('.jpg', '.jpeg', '.tiff')):
                    exif_dict = piexif.load(file_path)
                    geo = get_geo_data(exif_dict)
                    
                    if geo and source_geo:
                        distance = calculate_distance(source_geo, geo)
                        print(f"  Distance from source: {distance:.2f} km")
                    else:
                        print("  Distance: Unable to calculate (missing geo data)")
                else:
                    print("  Distance: Unable to calculate (unsupported file format for EXIF data)")
            except Exception as e:
                print(f"  Error processing EXIF data: {str(e)}")
            
            print()

def main():
    parser = argparse.ArgumentParser(description="Read and display tags from images in a folder and calculate distance from a source image.")
    parser.add_argument("folder_path", help="Path to the folder containing images")
    parser.add_argument("source_image", help="Path to the source image for distance calculation")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a valid directory")
        return

    if not os.path.isfile(args.source_image):
        print(f"Error: {args.source_image} is not a valid file")
        return

    source_exif = piexif.load(args.source_image)
    source_geo = get_geo_data(source_exif)
    if not source_geo:
        print("Warning: Source image does not contain geo data. Distances will not be calculated.")

    process_folder(args.folder_path, source_geo)

if __name__ == "__main__":
    main()