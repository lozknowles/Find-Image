import sys
import os
from PIL import Image
import piexif
from fractions import Fraction
import shutil
import argparse

def decimal_to_dms(decimal):
    """Convert decimal degrees to degrees, minutes, seconds"""
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = ((decimal - degrees) * 60 - minutes) * 60
    return (degrees, 1), (minutes, 1), (Fraction(int(seconds * 1000), 1000).limit_denominator())

def extract_geo_data(donor_image_path):
    try:
        exif_dict = piexif.load(donor_image_path)
        gps_ifd = exif_dict.get("GPS", {})
        
        lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
        lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
        alt = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
        alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef)
        
        if not all([lat, lat_ref, lon, lon_ref]):
            print(f"Error: {donor_image_path} does not contain complete GPS data.")
            sys.exit(1)
        
        return {
            'latitude': lat,
            'latitude_ref': lat_ref,
            'longitude': lon,
            'longitude_ref': lon_ref,
            'altitude': alt,
            'altitude_ref': alt_ref
        }
    except Exception as e:
        print(f"Error extracting GPS data from {donor_image_path}: {str(e)}")
        sys.exit(1)

def apply_geo_data(image_path, geo_data, tags, dry_run=False):
    try:
        if dry_run:
            print(f"Would add GPS data to: {os.path.basename(image_path)}")
            if tags:
                print(f"Would add tags: {tags}")
            return True

        img = Image.open(image_path)
        
        # For PNG files, convert to JPEG first
        if image_path.lower().endswith('.png'):
            jpg_path = os.path.splitext(image_path)[0] + '.jpg'
            img = img.convert('RGB')
            img.save(jpg_path, 'JPEG')
            image_path = jpg_path
            img = Image.open(image_path)
        
        # Load existing EXIF data or create new if not present
        try:
            exif_dict = piexif.load(img.info.get('exif', b''))
        except:
            exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
        
        # Create GPS IFD if it doesn't exist
        if "GPS" not in exif_dict:
            exif_dict["GPS"] = {}
        
        # Add GPS data
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = geo_data['latitude']
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = geo_data['latitude_ref']
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = geo_data['longitude']
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = geo_data['longitude_ref']
        
        if geo_data['altitude'] is not None:
            exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = geo_data['altitude']
            exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = geo_data['altitude_ref']
        
        # Add tags to EXIF metadata
        if tags:
            print(f"Adding tags: {tags}")
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = tags.encode('utf-8')
        
        # Remove problematic tags
        for ifd in ("0th", "Exif", "1st"):
            if ifd in exif_dict:
                for tag in list(exif_dict[ifd].keys()):
                    if isinstance(exif_dict[ifd][tag], int):
                        del exif_dict[ifd][tag]
        
        exif_bytes = piexif.dump(exif_dict)
        img.save(image_path, exif=exif_bytes)
        print(f"Processed: {os.path.basename(image_path)}")
        
        # Verify tags were added
        verify_tags(image_path)
        
        return True
    except Exception as e:
        print(f"Warning: Could not add data to {image_path}. Error: {str(e)}")
        return False

def verify_tags(image_path):
    try:
        exif_dict = piexif.load(image_path)
        if piexif.ImageIFD.ImageDescription in exif_dict["0th"]:
            tags = exif_dict["0th"][piexif.ImageIFD.ImageDescription].decode('utf-8')
            print(f"Verified tags for {os.path.basename(image_path)}: {tags}")
        else:
            print(f"No tags found for {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error verifying tags for {image_path}: {str(e)}")

def get_unique_filename(base_path):
    """Generate a unique filename by appending a number if the file already exists."""
    counter = 1
    file_path = base_path
    while os.path.exists(file_path):
        name, ext = os.path.splitext(base_path)
        file_path = f"{name}_{counter}{ext}"
        counter += 1
    return file_path

def process_images(donor_image_path, tags, toprocess_folder, processed_folder, dry_run=False, copy_mode=False):
    geo_data = extract_geo_data(donor_image_path)

    if not os.path.isdir(toprocess_folder):
        print(f"Error: input folder does not exist: {toprocess_folder}")
        sys.exit(1)

    if not os.path.exists(processed_folder):
        if dry_run:
            print(f"Would create folder: {processed_folder}")
        else:
            os.makedirs(processed_folder)

    for filename in os.listdir(toprocess_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(toprocess_folder, filename)
            output_path = os.path.join(processed_folder, filename)

            if apply_geo_data(input_path, geo_data, tags, dry_run=dry_run):
                unique_output_path = get_unique_filename(output_path)
                if dry_run:
                    action = "copy" if copy_mode else "move"
                    print(f"Would {action} to: {os.path.basename(unique_output_path)}")
                elif copy_mode:
                    shutil.copy2(input_path, unique_output_path)
                    print(f"Copied to: {os.path.basename(unique_output_path)}")
                else:
                    shutil.move(input_path, unique_output_path)
                    print(f"Moved to: {os.path.basename(unique_output_path)}")
            else:
                print(f"Skipped: {filename} (Unable to add GPS data)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geotag photos and add tags.")
    parser.add_argument("donor_image_path", help="Path to the donor image with GPS data")
    parser.add_argument("--tags", help="Comma-separated list of tags to add to the images")
    parser.add_argument("--input-folder", default="toprocess", help="Folder containing images to process")
    parser.add_argument("--output-folder", default="processed", help="Folder for processed images")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing metadata or moving files")
    parser.add_argument("--copy", action="store_true", help="Copy processed files instead of moving originals")
    args = parser.parse_args()

    print(f"Donor image: {args.donor_image_path}")
    print(f"Tags to be added: {args.tags}")

    process_images(
        args.donor_image_path,
        args.tags or "",
        args.input_folder,
        args.output_folder,
        dry_run=args.dry_run,
        copy_mode=args.copy,
    )
