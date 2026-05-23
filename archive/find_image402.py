import os
import sys
import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFont, ExifTags
import piexif
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import cv2
import uuid
import argparse

# Version of the script
VERSION = "4.0.2"
# Major changes in this version:
# - Corrected ResNet50 model configuration to use global average pooling
# - Fixed model loading and saving procedures
# - Clarified use of ImageNet pre-trained weights
# ... (previous version history)

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

# Load or download the ResNet50 model pre-trained on Google Landmarks Dataset
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

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

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

def calculate_confidence(cnn_similarity, geo_distance, geo_threshold):
    confidence = cnn_similarity
    print(f"Initial confidence (from similarity): {confidence:.4f}")
    print(f"Input geo_threshold: {geo_threshold:.4f}")
    
    if geo_distance is not None:
        print(f"Image distance: {geo_distance:.4f}")
        if geo_distance <= (geo_threshold + EPSILON):
            confidence += 0.1
            print(f"Adding 0.1 for being within threshold. New confidence: {confidence:.4f}")
        elif 0.5 < geo_distance <= 1.0:
            decrease = 0.05 + (geo_distance - 0.5) * 0.1
            confidence -= decrease
            print(f"Subtracting {decrease:.4f} for distance between 0.5 and 1.0 km. New confidence: {confidence:.4f}")
        elif geo_distance > 1.0:
            confidence -= 0.1
            print(f"Subtracting 0.1 for distance over 1.0 km. New confidence: {confidence:.4f}")
    else:
        print("No GPS data available.")

    confidence = max(min(confidence, 1.0), 0.0)
    print(f"Final confidence after capping between 0 and 1: {confidence:.4f}")
    return confidence

def find_similar_images_cnn(target_image_path, folder_path, similarity_threshold=0.7, geo_threshold=10):
    run_id = uuid.uuid4()
    print(f"Starting new run with ID: {run_id}")
    
    print(f"Extracting features from target image: {target_image_path}")
    target_features = extract_features(target_image_path)
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
            print(f"\nComparing with: {file_path}")
            try:
                current_features = extract_features(file_path)
                if current_features is None:
                    print(f"Error: Could not extract features from {file_path}")
                    continue

                cnn_similarity = calculate_similarity(target_features, current_features)
                
                if cnn_similarity >= similarity_threshold:
                    current_gps = extract_gps_info(file_path)
                    
                    geo_distance = geodesic(target_gps, current_gps).kilometers if target_gps and current_gps else None
                    confidence = calculate_confidence(cnn_similarity, geo_distance, geo_threshold)
                    
                    print(f"CNN Similarity: {cnn_similarity:.4f}")
                    print(f"Confidence: {confidence:.4f}")
                    if geo_distance is not None:
                        print(f"GPS Distance: {geo_distance:.4f} km")
                    else:
                        print("No GPS data available")

                    similar_images.append((file_path, file_name, confidence, cnn_similarity, current_gps, geo_distance))
                else:
                    print(f"Similarity {cnn_similarity:.4f} below threshold {similarity_threshold}, skipping")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

    similar_images.sort(key=lambda x: x[2], reverse=True)
    return similar_images

def create_montage(similar_images, target_image_path, similarity_threshold, geo_threshold):
    img_width, img_height = 300, 300
    cols = 6
    margin = 10
    font = ImageFont.load_default()

    total_images = len(similar_images) + 1  # +1 for the target image
    rows = (total_images + cols - 1) // cols  # Round up division

    montage_width = cols * (img_width + margin) + margin
    montage_height = rows * (img_height + margin + 75) + margin + 200 + 100  # Added extra 100px buffer

    montage = PilImage.new('RGB', (montage_width, montage_height), color='white')
    draw = ImageDraw.Draw(montage)

    header_text = f"Script: {os.path.basename(sys.argv[0])}\nVersion: {VERSION}\n"
    header_text += f"Image: {os.path.basename(target_image_path)}\nFolder: {os.path.basename(os.path.dirname(similar_images[0][0]))}\n"
    header_text += f"Similarity Threshold: {similarity_threshold}\nGeo Threshold: {geo_threshold} km\n"
    header_text += f"Model: ResNet50 (Google Landmarks Dataset)\n\nConfidence Calculation:\n"
    header_text += "1. Image similarity score\n2. If GPS data available:\n"
    header_text += "   a) Distance <= geo_threshold: +0.1 boost\n"
    header_text += "   b) 0.5 km < Distance <= 1.0 km: Graduated decrease\n"
    header_text += "   c) Distance > 1.0 km: -0.1 penalty\n"
    header_text += "3. Final confidence capped at 1.0"

    draw.text((margin, margin), header_text, fill='black', font=font)

    def add_image_to_montage(img_path, x, y, conf, sim, dist):
        img = PilImage.open(img_path).convert('RGB')
        img.thumbnail((img_width, img_height))
        montage.paste(img, (x, y))
        
        text_y = y + img_height + 5
        draw.text((x, text_y), f"Conf = {conf:.4f}", fill='black', font=font)
        draw.text((x, text_y + 15), f"Sim = {sim:.4f}", fill='black', font=font)
        if dist is not None:
            draw.text((x, text_y + 30), f"Dist: {dist:.4f} km", fill='black', font=font)
        else:
            draw.text((x, text_y + 30), "No GPS data", fill='black', font=font)

    # Add target image
    target_x, target_y = margin, 200  # Adjust based on header height
    add_image_to_montage(target_image_path, target_x, target_y, 1.0, 1.0, None)
    draw.text((target_x, target_y + img_height + 45), "Target Image", fill='black', font=font)

    # Add similar images
    for i, (img_path, _, confidence, similarity, _, geo_distance) in enumerate(similar_images):
        x = margin + (i % cols) * (img_width + margin)
        y = 200 + ((i // cols) + 1) * (img_height + margin + 75)  # Adjusted for additional text space
        add_image_to_montage(img_path, x, y, confidence, similarity, geo_distance)

    # Crop any unused white space at the bottom
    bbox = montage.getbbox()
    montage = montage.crop(bbox)

    version_number = VERSION.replace(".", "")
    montage_path = f'montage{version_number}.jpg'
    montage.save(montage_path)
    print(f"Montage saved as {montage_path}")

def main():
    parser = argparse.ArgumentParser(description='Find similar images based on CNN features and GPS data.')
    parser.add_argument('target_image', help='Path to the target image')
    parser.add_argument('folder', help='Path to the folder containing images to compare')
    parser.add_argument('similarity_threshold', type=float, help='Similarity threshold for image comparison')
    parser.add_argument('geo_threshold', type=float, help='Geographic distance threshold in km')
    args = parser.parse_args()

    similar_images = find_similar_images_cnn(args.target_image, args.folder, args.similarity_threshold, args.geo_threshold)
    
    if similar_images:
        create_montage(similar_images, args.target_image, args.similarity_threshold, args.geo_threshold)
    else:
        print("No similar images found.")

if __name__ == "__main__":
    main()