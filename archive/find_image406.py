import os
import sys
import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFont, ExifTags, ImageOps
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import uuid
import argparse
import piexif

# Version of the script
VERSION = "4.0.6"
# Changes in this version:
# - Improved sorting of similar images to prioritize higher similarity when confidence is equal
# - Updated montage creation to reflect the new sorting order
# - Ensured target image is always displayed first with correct similarity score
# - Fixed issue where images with equal confidence weren't properly ordered by similarity
# - Added optional tag search feature

# Installation instructions:
# 1. Ensure you have Python 3.7+ installed
# 2. Run the following commands to install required packages:
#    pip install tensorflow pillow numpy scikit-learn geopy opencv-python-headless
# 3. The script will automatically download and cache the ResNet50 model pre-trained on ImageNet

# Constants
EPSILON = 1e-6
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".landmark_model_cache")

# Ensure cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def get_landmark_model():
    model_path = os.path.join(MODEL_CACHE_DIR, "resnet50_imagenet.h5")
    if not os.path.exists(model_path):
        print("Downloading ResNet50 model pre-trained on ImageNet...")
        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Loading model from {model_path}")
        model = load_model(model_path)
    return model

model = get_landmark_model()

def open_image_with_correct_orientation(img_path):
    img = PilImage.open(img_path)
    # Apply EXIF orientation to ensure the image is correctly rotated
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')

def extract_features(img_path):
    try:
        img = open_image_with_correct_orientation(img_path)
        img = img.resize((224, 224))  # ResNet50 expects 224x224 images
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        
        # Extract tags from EXIF metadata
        tags = []
        try:
            exif_dict = piexif.load(img_path)
            if piexif.ImageIFD.ImageDescription in exif_dict["0th"]:
                tags_str = exif_dict["0th"][piexif.ImageIFD.ImageDescription].decode('utf-8')
                tags = [tag.strip() for tag in tags_str.split(',')]
            print(f"Tags found in {os.path.basename(img_path)}: {tags}")  # Debug print
        except Exception as e:
            print(f"Error reading EXIF metadata from {os.path.basename(img_path)}: {e}")
        
        return features.flatten(), tags
    except Exception as e:
        print(f"Error extracting features from {os.path.basename(img_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def calculate_similarity(features1, features2):
    return cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo':
                        gps_info = {}
                        for key in value:
                            decode = ExifTags.GPSTAGS.get(key, key)
                            gps_info[decode] = value[key]
                        lat = gps_info.get('GPSLatitude')
                        lon = gps_info.get('GPSLongitude')
                        if lat and lon:
                            lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                            lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                            if gps_info.get('GPSLatitudeRef') == 'S':
                                lat = -lat
                            if gps_info.get('GPSLongitudeRef') == 'W':
                                lon = -lon
                            return (lat, lon)
    except Exception as e:
        print(f"Error extracting GPS info from {img_path}: {e}")
    return None

def calculate_confidence(cnn_similarity, geo_distance, geo_threshold, matching_tags):
    confidence = cnn_similarity
    print(f"Initial confidence (from similarity): {confidence:.4f}")
    print(f"Input geo_threshold: {geo_threshold:.4f}")
    
    # Add confidence based on matching tags
    if matching_tags:
        if len(matching_tags) == 1:
            confidence += 0.075
            print(f"Adding 0.075 for 1 matching tag. New confidence: {confidence:.4f}")
        elif len(matching_tags) == 2:
            confidence += 0.1
            print(f"Adding 0.1 for 2 matching tags. New confidence: {confidence:.4f}")
        elif len(matching_tags) > 2:
            confidence += 0.1 + (len(matching_tags) - 2) * 0.05
            print(f"Adding {0.1 + (len(matching_tags) - 2) * 0.05:.4f} for {len(matching_tags)} matching tags. New confidence: {confidence:.4f}")

    if geo_distance is not None:
        print(f"Image distance: {geo_distance:.4f}")
        if geo_distance < 0.05:  # Less than 0.05 km
            confidence += 0.1
            print(f"Adding 0.1 for distance less than 0.05 km. New confidence: {confidence:.4f}")
        elif geo_distance < EPSILON:  # Locations are essentially the same
            confidence += 0.5
            print(f"Adding 0.5 for identical GPS location. New confidence: {confidence:.4f}")
        elif geo_distance <= (geo_threshold + EPSILON):
            confidence += 0.1
            print(f"Adding 0.1 for being within threshold. New confidence: {confidence:.4f}")
        elif geo_distance > 0.2:
            confidence = 0.0  # Set confidence to 0 for distances > 0.2 km
            print(f"Setting confidence to 0 for distance over 0.2 km. New confidence: {confidence:.4f}")
    else:
        print("No GPS data available. Confidence unchanged.")

    confidence = max(min(confidence, 1.0), 0.0)
    print(f"Final confidence after capping between 0 and 1: {confidence:.4f}")
    return confidence

def find_similar_images_cnn(target_image_path, folder_path, similarity_threshold=0.7, geo_threshold=10, search_tags=None):
    run_id = uuid.uuid4()
    print(f"Starting new run with ID: {run_id}")
    
    print(f"Extracting features from target image: {target_image_path}")
    target_features, _ = extract_features(target_image_path)
    if target_features is None:
        print("Error: Could not extract features from target image.")
        return []

    print(f"Attempting to extract GPS info from: {target_image_path}")
    target_gps = extract_gps_info(target_image_path)
    print(f"Extracted GPS info: {target_gps}")
    
    similar_images = []
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, file_name)
            print(f"\nComparing with: {file_name}")
            try:
                current_features, current_tags = extract_features(file_path)
                if current_features is None:
                    print(f"Error: Could not extract features from {file_name}")
                    continue

                cnn_similarity = calculate_similarity(target_features, current_features)
                
                if cnn_similarity >= similarity_threshold:
                    current_gps = extract_gps_info(file_path)
                    
                    geo_distance = geodesic(target_gps, current_gps).kilometers if target_gps and current_gps else None
                    
                    matching_tags = [tag for tag in search_tags if tag.lower() in [t.lower() for t in current_tags]] if search_tags else []
                    
                    confidence = calculate_confidence(cnn_similarity, geo_distance, geo_threshold, matching_tags)
                    
                    print(f"CNN Similarity: {cnn_similarity:.4f}")
                    print(f"Confidence: {confidence:.4f}")
                    if geo_distance is not None:
                        print(f"GPS Distance: {geo_distance:.4f} km")
                    else:
                        print("No GPS data available")
                    print(f"Matching tags: {matching_tags}")

                    if confidence > 0:  # Only add images with confidence > 0
                        similar_images.append((file_path, file_name, confidence, cnn_similarity, current_gps, geo_distance, matching_tags, current_tags))
                else:
                    print(f"Similarity {cnn_similarity:.4f} below threshold {similarity_threshold}, skipping")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

    # Replace the existing sorting line with this:
    similar_images.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return similar_images

def create_montage(similar_images, target_image_path, similarity_threshold, geo_threshold, search_tags):
    img_width, img_height = 300, 300
    cols = 6
    margin = 10
    font = ImageFont.load_default()

    total_images = len(similar_images) + 1  # +1 for the target image
    rows = (total_images + cols - 1) // cols  # Round up division

    montage_width = cols * (img_width + margin) + margin
    montage_height = rows * (img_height + margin + 105) + margin + 200  # Reduced height for condensed header

    montage = PilImage.new('RGB', (montage_width, montage_height), color='white')
    draw = ImageDraw.Draw(montage)

    # Condensed header information
    header_text = f"V{VERSION} | Folder: {os.path.basename(os.path.dirname(similar_images[0][0]))} | Sim: {similarity_threshold} | Geo: {geo_threshold}km | Tags: {', '.join(search_tags) if search_tags else 'None'}"
    draw.text((margin, margin), header_text, fill='black', font=font)

    # Additional information
    info_text = f"Model: ResNet50 (ImageNet) | Confidence: Similarity + GPS boost + 0.1 per matching tag (max 1.0)"
    draw.text((margin, margin + 20), info_text, fill='black', font=font)

    def add_image_to_montage(img_path, x, y, conf, sim, dist, matching_tags, all_tags):
        img = open_image_with_correct_orientation(img_path)
        img.thumbnail((img_width, img_height))
        montage.paste(img, (x, y))
        
        text_y = y + img_height + 5
        draw.text((x, text_y), f"Conf = {conf:.4f}", fill='black', font=font)
        draw.text((x, text_y + 15), f"Sim = {sim:.4f}", fill='black', font=font)
        if dist is not None:
            draw.text((x, text_y + 30), f"Dist: {dist:.4f} km", fill='black', font=font)
        else:
            draw.text((x, text_y + 30), "No GPS data", fill='black', font=font)
        draw.text((x, text_y + 45), f"Matching Tags: {len(matching_tags)}/{len(search_tags) if search_tags else 0}", fill='black', font=font)
        draw.text((x, text_y + 60), f"All Tags: {', '.join(all_tags[:3])}{'...' if len(all_tags) > 3 else ''}", fill='black', font=font)
        print(f"Adding to montage - {os.path.basename(img_path)}: Matching tags: {matching_tags}, All tags: {all_tags}")  # Debug print

    # Add target image
    target_x, target_y = margin, 60  # Adjusted y-coordinate for condensed header
    target_features, target_tags = extract_features(target_image_path)
    target_similarity = calculate_similarity(target_features, target_features)
    matching_target_tags = [tag for tag in search_tags if tag.lower() in [t.lower() for t in target_tags]] if search_tags else []
    add_image_to_montage(target_image_path, target_x, target_y, 1.0, target_similarity, None, matching_target_tags, target_tags)
    draw.text((target_x, target_y + img_height + 75), "Target Image", fill='black', font=font)

    # Add similar images
    for i, (img_path, _, confidence, similarity, _, geo_distance, matching_tags, all_tags) in enumerate(similar_images):
        x = margin + ((i + 1) % cols) * (img_width + margin)
        y = 60 + ((i + 1) // cols) * (img_height + margin + 105)  # Adjusted y-coordinate for condensed header
        add_image_to_montage(img_path, x, y, confidence, similarity, geo_distance, matching_tags, all_tags)

    # Crop any unused white space at the bottom
    bbox = montage.getbbox()
    montage = montage.crop(bbox)

    version_number = VERSION.replace(".", "")
    montage_path = f'montage{version_number}.jpg'
    montage.save(montage_path)
    print(f"Montage saved as {montage_path}")

def main():
    parser = argparse.ArgumentParser(description='Find similar images based on CNN features, GPS data, and tags.')
    parser.add_argument('target_image', help='Path to the target image')
    parser.add_argument('folder', help='Path to the folder containing images to compare')
    parser.add_argument('similarity_threshold', type=float, help='Similarity threshold for image comparison')
    parser.add_argument('geo_threshold', type=float, help='Geographic distance threshold in km')
    parser.add_argument('-tag', '--tags', help='Comma-separated list of tags to search for in image metadata')
    args = parser.parse_args()

    search_tags = [tag.strip() for tag in args.tags.split(',')] if args.tags else None

    similar_images = find_similar_images_cnn(args.target_image, args.folder, args.similarity_threshold, args.geo_threshold, search_tags)
    
    if similar_images:
        create_montage(similar_images, args.target_image, args.similarity_threshold, args.geo_threshold, search_tags)
    else:
        print("No similar images found.")

if __name__ == "__main__":
    main()