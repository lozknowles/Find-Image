import os
import sys
import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFont, ExifTags
import piexif
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from sklearn.preprocessing import normalize
import torch
import cv2
import uuid
import argparse

# Version of the script
VERSION = "3.0.34"
# Changes in this version:
# - Fixed precision issue in distance threshold comparison
# - Updated montage to display distances with higher precision (4 decimal places)
# - Added epsilon value to account for floating-point precision in distance comparisons
# Changes in 3.0.33:
# - Added detailed debug output for confidence calculation process
# - Ensured correct application of confidence boost for images within geo_threshold
# Changes in 3.0.32:
# - Fixed critical bug in confidence calculation for images within the geo_threshold
# - Ensured all images within the geo_threshold receive the appropriate confidence boost
# Changes in 3.0.31:
# - Implemented graduated confidence decrease for images between 0.5km and 1.0km away
# - Added maximum confidence decrease for images over 1.0km away
# - Ensured confidence always stays between 0 and 1
# Changes in 3.0.30:
# - Fixed confidence calculation to properly boost scores for images exactly at the geo_threshold
# - Ensured consistent application of distance-based confidence boost
# Changes in 3.0.29:
# - Fixed confidence calculation to properly boost scores for images within or equal to the geo_threshold
# - Ensured all images within distance threshold receive at least a small confidence boost
# Changes in 3.0.28:
# - Fixed confidence calculation to not boost scores for images without GPS data
# - Ensured confidence is never higher than similarity when there's no GPS data or object similarity
# Changes in 3.0.27:
# - Fixed confidence calculation to not boost scores for images beyond the geo_threshold
# - Ensured images without GPS data are not penalized in confidence calculation
# Changes in 3.0.26:
# - Aligned with version 2.0.1's behavior
# - Added L2 normalization to feature extraction
# - Updated confidence calculation
# - Ensured consistent MobileNetV2 configuration
# - Fixed issues in create_montage function:
#   * Ensured target image is always included
#   * Addressed layout gaps and image sizing inconsistencies
# Changes in 3.0.25:
# - Fixed confidence calculation to properly increase when distance is within threshold
# - Improved object detection error handling and logging
# - Added more detailed YOLO model initialization process
# - Updated comments to reflect recent changes
# Changes in 3.0.24:
# - Fixed syntax error related to incomplete try-except block
# - Removed duplicate import statements and version history
# - Ensured all import statements are at the top of the file
# - Added missing function definitions
# - Fixed issues with create_montage function
# ... (rest of the version history)

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load YOLO model with error handling
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.eval()  # Set the model to evaluation mode
    use_yolo = True
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Continuing without object detection...")
    use_yolo = False

def extract_features(img_path):
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)  # Add L2 normalization
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

def detect_objects(img_path):
    global use_yolo
    if use_yolo:
        try:
            img = cv2.imread(img_path)
            results = yolo_model(img)
            objects = results.pandas().xyxy[0]['name'].tolist()
            print(f"Detected objects: {objects}")
            return objects
        except Exception as e:
            print(f"Error detecting objects in {img_path}: {e}")
            print("Detailed error:")
            import traceback
            traceback.print_exc()
            use_yolo = False  # Disable YOLO for future calls if an error occurs
    else:
        print("YOLO object detection is disabled.")
    return []

def calculate_object_similarity(objects1, objects2):
    set1 = set(objects1)
    set2 = set(objects2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def open_image_with_orientation(image_path):
    try:
        image = PilImage.open(image_path)
        
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = dict(image._getexif().items())
            
            if ExifTags.TAGS.get('Orientation', 274) in exif:
                orientation = exif[ExifTags.TAGS.get('Orientation', 274)]
                
                if orientation == 2:
                    image = image.transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180)
                elif orientation == 4:
                    image = image.rotate(180).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    image = image.rotate(-90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    image = image.rotate(-90, expand=True)
                elif orientation == 7:
                    image = image.rotate(90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        
        return image
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

def calculate_confidence(cnn_similarity, geo_distance, geo_threshold, object_similarity):
    confidence = cnn_similarity
    print(f"Initial confidence (from similarity): {confidence:.4f}")
    print(f"Input geo_threshold: {geo_threshold:.4f}")
    print(f"Image distance: {geo_distance:.4f}")

    EPSILON = 1e-6  # Small value to account for floating-point precision

    if geo_distance is not None:
        if geo_distance <= (geo_threshold + EPSILON):
            if cnn_similarity > 0.48 and geo_distance < 0.1:
                confidence += 0.3
                print(f"Adding 0.3 for high similarity and very close distance. New confidence: {confidence:.4f}")
            else:
                confidence += 0.05
                print(f"Adding 0.05 for being within threshold. New confidence: {confidence:.4f}")
        elif 0.5 < geo_distance <= 1.0:
            decrease = 0.05 + (geo_distance - 0.5) * 0.1
            confidence -= decrease
            print(f"Subtracting {decrease:.4f} for distance between 0.5 and 1.0 km. New confidence: {confidence:.4f}")
        elif geo_distance > 1.0:
            confidence -= 0.1
            print(f"Subtracting 0.1 for distance over 1.0 km. New confidence: {confidence:.4f}")
    else:
        print("No GPS data available.")

    if object_similarity > 0.5:
        confidence += 0.2
        print(f"Adding 0.2 for high object similarity. New confidence: {confidence:.4f}")
    elif object_similarity > 0.3:
        confidence += 0.1
        print(f"Adding 0.1 for medium object similarity. New confidence: {confidence:.4f}")

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

    print(f"Detecting objects in target image: {target_image_path}")
    target_objects = detect_objects(target_image_path)
    print(f"Detected objects in target image: {target_objects}")

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
                    current_objects = detect_objects(file_path)
                    object_similarity = calculate_object_similarity(target_objects, current_objects)
                    current_gps = extract_gps_info(file_path)
                    
                    confidence = calculate_confidence(cnn_similarity, geodesic(target_gps, current_gps).kilometers if target_gps and current_gps else None, geo_threshold, object_similarity)
                    
                    print(f"CNN Similarity: {cnn_similarity:.4f}")
                    print(f"Object Similarity: {object_similarity:.4f}")
                    print(f"Confidence: {confidence:.4f}")
                    if current_gps is not None:
                        print(f"GPS Distance: {geodesic(target_gps, current_gps).kilometers:.4f} km")
                    else:
                        print("No GPS data available")

                    similar_images.append((file_path, file_name, confidence, cnn_similarity, object_similarity, current_gps, geodesic(target_gps, current_gps).kilometers if target_gps and current_gps else None, None, None, None))
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
    montage_height = rows * (img_height + margin + 75) + margin + 200  # Adjusted for additional text space

    montage = PilImage.new('RGB', (montage_width, montage_height), color='white')
    draw = ImageDraw.Draw(montage)

    header_text = f"Script: {os.path.basename(sys.argv[0])}\nVersion: {VERSION}\n"
    header_text += f"Image: {os.path.basename(target_image_path)}\nFolder: {os.path.basename(os.path.dirname(similar_images[0][0]))}\n"
    header_text += f"Similarity Threshold: {similarity_threshold}\nGeo Threshold: {geo_threshold} km\n"
    header_text += f"Model: MobileNetV2\n\nConfidence Calculation:\n"
    header_text += "1. Image similarity score\n2. If GPS data available:\n"
    header_text += "   a) similarity > 0.48 and Distance < 0.1km: +0.3 boost\n"
    header_text += "   b) Distance <= geo_threshold: +0.05 boost\n"
    header_text += "   c) Distance > geo_threshold: No penalty\n"
    header_text += "3. Object similarity > 0.5: +0.2 boost\n"
    header_text += "4. Object similarity > 0.3: +0.1 boost\n"
    header_text += "5. Final confidence capped at 1.0"

    draw.text((margin, margin), header_text, fill='black', font=font)

    def add_image_to_montage(img_path, x, y, conf, sim, obj, dist):
        img = open_image_with_orientation(img_path)
        img = img.convert('RGB')
        img.thumbnail((img_width, img_height))
        montage.paste(img, (x, y))
        
        text_y = y + img_height + 5
        draw.text((x, text_y), f"Conf = {conf:.4f}", fill='black', font=font)
        draw.text((x, text_y + 15), f"Sim = {sim:.4f}", fill='black', font=font)
        draw.text((x, text_y + 30), f"Obj = {obj:.4f}", fill='black', font=font)
        if dist is not None:
            draw.text((x, text_y + 45), f"Dist: {dist:.4f} km", fill='black', font=font)
        else:
            draw.text((x, text_y + 45), "No GPS data", fill='black', font=font)

    # Add target image
    target_x, target_y = margin, 200  # Adjust based on header height
    add_image_to_montage(target_image_path, target_x, target_y, 1.0, 1.0, 1.0, None)
    draw.text((target_x, target_y + img_height + 65), "Target Image", fill='black', font=font)

    # Add similar images
    for i, (img_path, _, confidence, similarity, object_similarity, _, geo_distance, _, _, _) in enumerate(similar_images, start=1):
        x = margin + (i % cols) * (img_width + margin)
        y = 200 + (i // cols + 1) * (img_height + margin + 75)  # Adjusted for additional text space
        add_image_to_montage(img_path, x, y, confidence, similarity, object_similarity, geo_distance)

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