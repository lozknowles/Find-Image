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
VERSION = "3.0.24"
# Changes in this version:
# - Fixed syntax error related to incomplete try-except block
# - Removed duplicate import statements and version history
# - Ensured all import statements are at the top of the file
# Changes in 3.0.23:
# - Updated montage creation to display object similarity scores
# - Adjusted montage layout to accommodate additional information
# - Updated header information to include object similarity in confidence calculation
# Changes in 3.0.22:
# - Implemented object recognition in similarity calculations
# - Added object similarity score to confidence calculation
# - Updated confidence calculation to consider object similarity
# Changes in 3.0.21:
# - Revised confidence calculation to prioritize similarity over GPS data
# - GPS weighting now only applied when distance is within the specified threshold
# - Adjusted confidence boost tiers based on similarity score
# Changes in 3.0.20:
# - Fixed montage creation to display all similar images without cropping
# - Adjusted montage height calculation to accommodate all rows of images
# Changes in 3.0.19:
# - Fixed error in create_montage function when handling images without GPS data
# - Improved error handling for missing GPS information
# Changes in 3.0.18:
# - Fixed montage layout to display 6 images per row
# - Corrected issue with blank first image in the montage
# - Adjusted montage to show up to 23 similar images (6 columns * 4 rows - 1 target image)
# Changes in 3.0.17:
# - Modified feature extraction to be more color-agnostic
# - Adjusted similarity threshold handling to potentially include more diverse images
# - Added more detailed logging for image exclusion reasons
# Changes in 3.0.16:
# - Fixed GPS data extraction and display
# - Corrected confidence calculation to properly account for GPS data
# - Added more detailed logging for GPS data extraction
# Changes in 3.0.15:
# - Fixed incorrect confidence calculation when no GPS or object data is available
# - Ensured confidence score does not exceed similarity score without additional data
# Changes in 3.0.14:
# - Fixed issue with scoring data not being visible in the montage
# - Adjusted montage layout to properly display confidence, similarity, object, and GPS scores
# Changes in 3.0.13:
# - Restored scoring data (confidence, similarity, object, GPS) beneath each image in the montage
# - Fixed image rotation issues by respecting EXIF orientation
# Changes in 3.0.12:
# - Fixed similarity threshold enforcement in find_similar_images_cnn function
# - Removed object recognition from similarity and confidence calculations
# - Updated create_montage function to reflect changes in similarity calculation
# Changes in 3.0.11:
# - Reverted and updated create_montage function to match the original layout and information display
# Changes in 3.0.10:
# - Reverted create_montage function to a more traditional layout for consistency with previous versions
# Changes in 3.0.9:
# - Added create_montage function to generate a visual representation of similar images
# Changes in 3.0.8:
# - Improved error handling and logging in find_similar_images_cnn function
# - Added checks for None values in feature extraction and similarity calculation
# - Enhanced debugging output for each image comparison
# Changes in 3.0.7:
# - Fixed issue with undefined target_objects in find_similar_images_cnn function
# - Added more detailed logging for object detection in target image
# Changes in 3.0.6:
# - Added missing extract_gps_info function
# - Ensured all necessary functions are defined and in the correct order
# Changes in 3.0.5:
# - Added more detailed logging throughout the script
# - Implemented a unique run ID for each execution
# - Added a --force-refresh command-line option (functionality to be implemented)
# - Improved error handling and reporting
# - Fixed the detect_objects function to set the confidence threshold correctly
# Changes in 3.0.4:
# - Added error handling for missing dependencies
# - Provided fallback option when YOLO model fails to load
# - Added missing functions: open_image_with_orientation, extract_gps_info, calculate_geo_distance
# - Added fallback detection method using edge detection
# - Modified detect_objects function to use a lower confidence threshold and include the fallback method
# - Updated calculate_object_similarity to handle the new object detection format
# - Added more detailed debug prints in the find_similar_images_cnn function
# Changes in 3.0.3:
# - Added individual contribution values (CNN, Object, GPS) to the montage output for each image
# Changes in 3.0.2:
# - Added model information (MobileNetV2 and YOLOv5) to the montage printout
# Changes from 3.0.1:
# - Added YOLO object detection for semantic understanding
# - Implemented ensemble method with weighted scoring
# - Created composite confidence score combining CNN features, object detection, and GPS data
# - Renamed script to find_image30.py for isolated testing against find_image20.py
# Inherited from find_image20.py (2.0.1):
# - Use of tensorflow.keras.utils.img_to_array
# - Improved error handling in feature extraction
# - Specified input_shape for MobileNetV2

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load YOLO model with error handling
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    use_yolo = True
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Continuing without object detection...")
    use_yolo = False

# Function to open image with correct orientation
def open_image_with_orientation(img_path):
    try:
        img = PilImage.open(img_path)
        
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.ImageIFD.Orientation in exif_dict["0th"]:
                orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
                if orientation == 2:
                    img = img.transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.rotate(180).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img = img.rotate(-90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    
        return img
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

# Function to extract GPS information from image
def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = {}
                        for key in value:
                            decode = ExifTags.GPSTAGS.get(key, key)
                            gps_info[decode] = value[key]
                        
                        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                            lat = gps_info["GPSLatitude"]
                            lon = gps_info["GPSLongitude"]
                            lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                            lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                            if gps_info["GPSLatitudeRef"] == "S":
                                lat = -lat
                            if gps_info["GPSLongitudeRef"] == "W":
                                lon = -lon
                            return (lat, lon)
    except Exception as e:
        print(f"Error extracting GPS info from {img_path}: {e}")
    return None

# Function to calculate geographical distance
def calculate_geo_distance(gps1, gps2):
    if gps1 and gps2:
        return geodesic(gps1, gps2).kilometers
    return None

def fallback_detection(img_path):
    """Fallback method to detect significant structures in an image using edge detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return len(significant_contours) > 0

def detect_objects(img_path):
    """
    Detect objects in an image using YOLO model with a lower confidence threshold.
    Falls back to edge detection if no objects are found.
    """
    print(f"Detecting objects in: {img_path}")
    if not use_yolo:
        return []
    img = cv2.imread(img_path)
    results = yolo_model(img)
    results.conf = 0.25  # Set confidence threshold
    detections = results.pandas().xyxy[0]
    objects = [f"{row['name']}:{row['confidence']:.2f}" for _, row in detections.iterrows()]
    if not objects and fallback_detection(img_path):
        objects = ['unknown_structure:0.50']
    print(f"Detected objects: {objects}")
    return objects

def calculate_object_similarity(objects1, objects2):
    """
    Calculate similarity between two sets of detected objects.
    """
    if not objects1 or not objects2:
        return 0.0
    set1 = set(obj.split(':')[0] for obj in objects1)
    set2 = set(obj.split(':')[0] for obj in objects2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    print(f"Object similarity: {similarity}")
    return similarity

# Function to extract features using MobileNetV2
def extract_features(img_path):
    print(f"Extracting features from: {img_path}")
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.convert('L')  # Convert to grayscale
        img = img.convert('RGB')  # Convert back to RGB (3 channels, but grayscale)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)
        print(f"Feature shape: {features.shape}")
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def calculate_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    print(f"Calculated similarity: {similarity}")
    return similarity

# Updated find_similar_images_cnn function
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
                
                # Only proceed if the similarity meets or exceeds the threshold
                if cnn_similarity >= similarity_threshold:
                    current_objects = detect_objects(file_path)
                    object_similarity = calculate_object_similarity(target_objects, current_objects)
                    current_gps = extract_gps_info(file_path)
                    
                    confidence = cnn_similarity  # Start with the similarity score
                    geo_distance = None

                    if target_gps and current_gps:
                        geo_distance = geodesic(target_gps, current_gps).km
                        if geo_distance <= geo_threshold:
                            if cnn_similarity < 0.48:
                                confidence += 0.3
                            elif 0.48 <= cnn_similarity < 0.6:
                                confidence += 0.2
                            else:
                                confidence += 0.1
                    
                    # Add object similarity to confidence calculation
                    if object_similarity > 0.5:
                        confidence += 0.2
                    elif object_similarity > 0.3:
                        confidence += 0.1
                    
                    confidence = min(confidence, 1.0)
                    
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
VERSION = "3.0.24"
# Changes in this version:
# - Fixed syntax error related to incomplete try-except block
# - Ensured all import statements are at the top of the file
# Changes in 3.0.23:
# - Updated montage creation to display object similarity scores
# - Adjusted montage layout to accommodate additional information
# - Updated header information to include object similarity in confidence calculation
# Changes in 3.0.22:
# - Implemented object recognition in similarity calculations
# - Added object similarity score to confidence calculation
# - Updated confidence calculation to consider object similarity
# - Preserved all existing functionality and comments
# Changes in 3.0.21:
# - Revised confidence calculation to prioritize similarity over GPS data
# - GPS weighting now only applied when distance is within the specified threshold
# - Adjusted confidence boost tiers based on similarity score
# Changes in 3.0.20:
# - Fixed montage creation to display all similar images without cropping
# - Adjusted montage height calculation to accommodate all rows of images
# Changes in 3.0.19:
# - Fixed error in create_montage function when handling images without GPS data
# - Improved error handling for missing GPS information
# Changes in 3.0.18:
# - Fixed montage layout to display 6 images per row
# - Corrected issue with blank first image in the montage
# - Adjusted montage to show up to 23 similar images (6 columns * 4 rows - 1 target image)
# Changes in 3.0.17:
# - Modified feature extraction to be more color-agnostic
# - Adjusted similarity threshold handling to potentially include more diverse images
# - Added more detailed logging for image exclusion reasons
# Changes in 3.0.16:
# - Fixed GPS data extraction and display
# - Corrected confidence calculation to properly account for GPS data
# - Added more detailed logging for GPS data extraction
# Changes in 3.0.15:
# - Fixed incorrect confidence calculation when no GPS or object data is available
# - Ensured confidence score does not exceed similarity score without additional data
# Changes in 3.0.14:
# - Fixed issue with scoring data not being visible in the montage
# - Adjusted montage layout to properly display confidence, similarity, object, and GPS scores
# Changes in 3.0.13:
# - Restored scoring data (confidence, similarity, object, GPS) beneath each image in the montage
# - Fixed image rotation issues by respecting EXIF orientation
# Changes in 3.0.12:
# - Fixed similarity threshold enforcement in find_similar_images_cnn function
# - Removed object recognition from similarity and confidence calculations
# - Updated create_montage function to reflect changes in similarity calculation
# Changes in 3.0.11:
# - Reverted and updated create_montage function to match the original layout and information display
# Changes in 3.0.10:
# - Reverted create_montage function to a more traditional layout for consistency with previous versions
# Changes in 3.0.9:
# - Added create_montage function to generate a visual representation of similar images
# Changes in 3.0.8:
# - Improved error handling and logging in find_similar_images_cnn function
# - Added checks for None values in feature extraction and similarity calculation
# - Enhanced debugging output for each image comparison
# Changes in 3.0.7:
# - Fixed issue with undefined target_objects in find_similar_images_cnn function
# - Added more detailed logging for object detection in target image
# Changes in 3.0.6:
# - Added missing extract_gps_info function
# - Ensured all necessary functions are defined and in the correct order
# Changes in 3.0.5:
# - Added more detailed logging throughout the script
# - Implemented a unique run ID for each execution
# - Added a --force-refresh command-line option (functionality to be implemented)
# - Improved error handling and reporting
# - Fixed the detect_objects function to set the confidence threshold correctly
# Changes in 3.0.4:
# - Added error handling for missing dependencies
# - Provided fallback option when YOLO model fails to load
# - Added missing functions: open_image_with_orientation, extract_gps_info, calculate_geo_distance
# - Added fallback detection method using edge detection
# - Modified detect_objects function to use a lower confidence threshold and include the fallback method
# - Updated calculate_object_similarity to handle the new object detection format
# - Added more detailed debug prints in the find_similar_images_cnn function
# Changes in 3.0.3:
# - Added individual contribution values (CNN, Object, GPS) to the montage output for each image
# Changes in 3.0.2:
# - Added model information (MobileNetV2 and YOLOv5) to the montage printout
# Changes from 3.0.1:
# - Added YOLO object detection for semantic understanding
# - Implemented ensemble method with weighted scoring
# - Created composite confidence score combining CNN features, object detection, and GPS data
# - Renamed script to find_image30.py for isolated testing against find_image20.py
# Inherited from find_image20.py (2.0.1):
# - Use of tensorflow.keras.utils.img_to_array
# - Improved error handling in feature extraction
# - Specified input_shape for MobileNetV2

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load YOLO model with error handling
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    use_yolo = True
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Continuing without object detection...")
    use_yolo = False

# Function to open image with correct orientation
def open_image_with_orientation(img_path):
    try:
        img = PilImage.open(img_path)
        
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.ImageIFD.Orientation in exif_dict["0th"]:
                orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
                if orientation == 2:
                    img = img.transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.rotate(180).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img = img.rotate(-90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    
        return img
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

# Function to extract GPS information from image
def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = {}
                        for key in value:
                            decode = ExifTags.GPSTAGS.get(key, key)
                            gps_info[decode] = value[key]
                        
                        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                            lat = gps_info["GPSLatitude"]
                            lon = gps_info["GPSLongitude"]
                            lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                            lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                            if gps_info["GPSLatitudeRef"] == "S":
                                lat = -lat
                            if gps_info["GPSLongitudeRef"] == "W":
                                lon = -lon
                            return (lat, lon)
    except Exception as e:
        print(f"Error extracting GPS info from {img_path}: {e}")
    return None

# Function to calculate geographical distance
def calculate_geo_distance(gps1, gps2):
    if gps1 and gps2:
        return geodesic(gps1, gps2).kilometers
    return None

def fallback_detection(img_path):
    """Fallback method to detect significant structures in an image using edge detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return len(significant_contours) > 0

def detect_objects(img_path):
    """
    Detect objects in an image using YOLO model with a lower confidence threshold.
    Falls back to edge detection if no objects are found.
    """
    print(f"Detecting objects in: {img_path}")
    if not use_yolo:
        return []
    img = cv2.imread(img_path)
    results = yolo_model(img)
    results.conf = 0.25  # Set confidence threshold
    detections = results.pandas().xyxy[0]
    objects = [f"{row['name']}:{row['confidence']:.2f}" for _, row in detections.iterrows()]
    if not objects and fallback_detection(img_path):
        objects = ['unknown_structure:0.50']
    print(f"Detected objects: {objects}")
    return objects

def calculate_object_similarity(objects1, objects2):
    """
    Calculate similarity between two sets of detected objects.
    """
    if not objects1 or not objects2:
        return 0.0
    set1 = set(obj.split(':')[0] for obj in objects1)
    set2 = set(obj.split(':')[0] for obj in objects2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    print(f"Object similarity: {similarity}")
    return similarity

# Function to extract features using MobileNetV2
def extract_features(img_path):
    print(f"Extracting features from: {img_path}")
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.convert('L')  # Convert to grayscale
        img = img.convert('RGB')  # Convert back to RGB (3 channels, but grayscale)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)
        print(f"Feature shape: {features.shape}")
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def calculate_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    print(f"Calculated similarity: {similarity}")
    return similarity

# Updated find_similar_images_cnn function
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
                
                # Only proceed if the similarity meets or exceeds the threshold
                if cnn_similarity >= similarity_threshold:
                    current_objects = detect_objects(file_path)
                    object_similarity = calculate_object_similarity(target_objects, current_objects)
                    current_gps = extract_gps_info(file_path)
                    
                    confidence = cnn_similarity  # Start with the similarity score
                    geo_distance = None

                    if target_gps and current_gps:
                        geo_distance = geodesic(target_gps, current_gps).km
                        if geo_distance <= geo_threshold:
                            if cnn_similarity < 0.48:
                                confidence += 0.3
                            elif 0.48 <= cnn_similarity < 0.6:
                                confidence += 0.2
                            else:
                                confidence += 0.1
                    
                    # Add object similarity to confidence calculation
                    if object_similarity > 0.5:
                        confidence += 0.2
                    elif object_similarity > 0.3:
                        confidence += 0.1
                    
                    confidence = min(confidence, 1.0)
                    
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
VERSION = "3.0.24"
# Changes in this version:
# - Fixed syntax error related to incomplete try-except block
# - Ensured all import statements are at the top of the file
# Changes in 3.0.23:
# - Updated montage creation to display object similarity scores
# - Adjusted montage layout to accommodate additional information
# - Updated header information to include object similarity in confidence calculation
# Changes in 3.0.22:
# - Implemented object recognition in similarity calculations
# - Added object similarity score to confidence calculation
# - Updated confidence calculation to consider object similarity
# - Preserved all existing functionality and comments
# Changes in 3.0.21:
# - Revised confidence calculation to prioritize similarity over GPS data
# - GPS weighting now only applied when distance is within the specified threshold
# - Adjusted confidence boost tiers based on similarity score
# Changes in 3.0.20:
# - Fixed montage creation to display all similar images without cropping
# - Adjusted montage height calculation to accommodate all rows of images
# Changes in 3.0.19:
# - Fixed error in create_montage function when handling images without GPS data
# - Improved error handling for missing GPS information
# Changes in 3.0.18:
# - Fixed montage layout to display 6 images per row
# - Corrected issue with blank first image in the montage
# - Adjusted montage to show up to 23 similar images (6 columns * 4 rows - 1 target image)
# Changes in 3.0.17:
# - Modified feature extraction to be more color-agnostic
# - Adjusted similarity threshold handling to potentially include more diverse images
# - Added more detailed logging for image exclusion reasons
# Changes in 3.0.16:
# - Fixed GPS data extraction and display
# - Corrected confidence calculation to properly account for GPS data
# - Added more detailed logging for GPS data extraction
# Changes in 3.0.15:
# - Fixed incorrect confidence calculation when no GPS or object data is available
# - Ensured confidence score does not exceed similarity score without additional data
# Changes in 3.0.14:
# - Fixed issue with scoring data not being visible in the montage
# - Adjusted montage layout to properly display confidence, similarity, object, and GPS scores
# Changes in 3.0.13:
# - Restored scoring data (confidence, similarity, object, GPS) beneath each image in the montage
# - Fixed image rotation issues by respecting EXIF orientation
# Changes in 3.0.12:
# - Fixed similarity threshold enforcement in find_similar_images_cnn function
# - Removed object recognition from similarity and confidence calculations
# - Updated create_montage function to reflect changes in similarity calculation
# Changes in 3.0.11:
# - Reverted and updated create_montage function to match the original layout and information display
# Changes in 3.0.10:
# - Reverted create_montage function to a more traditional layout for consistency with previous versions
# Changes in 3.0.9:
# - Added create_montage function to generate a visual representation of similar images
# Changes in 3.0.8:
# - Improved error handling and logging in find_similar_images_cnn function
# - Added checks for None values in feature extraction and similarity calculation
# - Enhanced debugging output for each image comparison
# Changes in 3.0.7:
# - Fixed issue with undefined target_objects in find_similar_images_cnn function
# - Added more detailed logging for object detection in target image
# Changes in 3.0.6:
# - Added missing extract_gps_info function
# - Ensured all necessary functions are defined and in the correct order
# Changes in 3.0.5:
# - Added more detailed logging throughout the script
# - Implemented a unique run ID for each execution
# - Added a --force-refresh command-line option (functionality to be implemented)
# - Improved error handling and reporting
# - Fixed the detect_objects function to set the confidence threshold correctly
# Changes in 3.0.4:
# - Added error handling for missing dependencies
# - Provided fallback option when YOLO model fails to load
# - Added missing functions: open_image_with_orientation, extract_gps_info, calculate_geo_distance
# - Added fallback detection method using edge detection
# - Modified detect_objects function to use a lower confidence threshold and include the fallback method
# - Updated calculate_object_similarity to handle the new object detection format
# - Added more detailed debug prints in the find_similar_images_cnn function
# Changes in 3.0.3:
# - Added individual contribution values (CNN, Object, GPS) to the montage output for each image
# Changes in 3.0.2:
# - Added model information (MobileNetV2 and YOLOv5) to the montage printout
# Changes from 3.0.1:
# - Added YOLO object detection for semantic understanding
# - Implemented ensemble method with weighted scoring
# - Created composite confidence score combining CNN features, object detection, and GPS data
# - Renamed script to find_image30.py for isolated testing against find_image20.py
# Inherited from find_image20.py (2.0.1):
# - Use of tensorflow.keras.utils.img_to_array
# - Improved error handling in feature extraction
# - Specified input_shape for MobileNetV2

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load YOLO model with error handling
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    use_yolo = True
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Continuing without object detection...")
    use_yolo = False

# Function to open image with correct orientation
def open_image_with_orientation(img_path):
    try:
        img = PilImage.open(img_path)
        
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.ImageIFD.Orientation in exif_dict["0th"]:
                orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
                if orientation == 2:
                    img = img.transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.rotate(180).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img = img.rotate(-90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    
        return img
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

# Function to extract GPS information from image
def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = {}
                        for key in value:
                            decode = ExifTags.GPSTAGS.get(key, key)
                            gps_info[decode] = value[key]
                        
                        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                            lat = gps_info["GPSLatitude"]
                            lon = gps_info["GPSLongitude"]
                            lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                            lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                            if gps_info["GPSLatitudeRef"] == "S":
                                lat = -lat
                            if gps_info["GPSLongitudeRef"] == "W":
                                lon = -lon
                            return (lat, lon)
    except Exception as e:
        print(f"Error extracting GPS info from {img_path}: {e}")
    return None

# Function to calculate geographical distance
def calculate_geo_distance(gps1, gps2):
    if gps1 and gps2:
        return geodesic(gps1, gps2).kilometers
    return None

def fallback_detection(img_path):
    """Fallback method to detect significant structures in an image using edge detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return len(significant_contours) > 0

def detect_objects(img_path):
    """
    Detect objects in an image using YOLO model with a lower confidence threshold.
    Falls back to edge detection if no objects are found.
    """
    print(f"Detecting objects in: {img_path}")
    if not use_yolo:
        return []
    img = cv2.imread(img_path)
    results = yolo_model(img)
    results.conf = 0.25  # Set confidence threshold
    detections = results.pandas().xyxy[0]
    objects = [f"{row['name']}:{row['confidence']:.2f}" for _, row in detections.iterrows()]
    if not objects and fallback_detection(img_path):
        objects = ['unknown_structure:0.50']
    print(f"Detected objects: {objects}")
    return objects

def calculate_object_similarity(objects1, objects2):
    """
    Calculate similarity between two sets of detected objects.
    """
    if not objects1 or not objects2:
        return 0.0
    set1 = set(obj.split(':')[0] for obj in objects1)
    set2 = set(obj.split(':')[0] for obj in objects2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    print(f"Object similarity: {similarity}")
    return similarity

# Function to extract features using MobileNetV2
def extract_features(img_path):
    print(f"Extracting features from: {img_path}")
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.convert('L')  # Convert to grayscale
        img = img.convert('RGB')  # Convert back to RGB (3 channels, but grayscale)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)
        print(f"Feature shape: {features.shape}")
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def calculate_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    print(f"Calculated similarity: {similarity}")
    return similarity

# Updated find_similar_images_cnn function
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
                
                # Only proceed if the similarity meets or exceeds the threshold
                if cnn_similarity >= similarity_threshold:
                    current_objects = detect_objects(file_path)
                    object_similarity = calculate_object_similarity(target_objects, current_objects)
                    current_gps = extract_gps_info(file_path)
                    
                    confidence = cnn_similarity  # Start with the similarity score
                    geo_distance = None

                    if target_gps and current_gps:
                        geo_distance = geodesic(target_gps, current_gps).km
                        if geo_distance <= geo_threshold:
                            if cnn_similarity < 0.48:
                                confidence += 0.3
                            elif 0.48 <= cnn_similarity < 0.6:
                                confidence += 0.2
                            else:
                                confidence += 0.1
                    
                    # Add object similarity to confidence calculation
                    if object_similarity > 0.5:
                        confidence += 0.2
                    elif object_similarity > 0.3:
                        confidence += 0.1
                    
                    confidence = min(confidence, 1.0)
                    
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
VERSION = "3.0.23"
# Changes in this version:
# - Updated montage creation to display object similarity scores
# - Adjusted montage layout to accommodate additional information
# - Updated header information to include object similarity in confidence calculation
# Changes in 3.0.22:
# - Implemented object recognition in similarity calculations
# - Added object similarity score to confidence calculation
# - Updated confidence calculation to consider object similarity
# - Preserved all existing functionality and comments
# Changes in 3.0.21:
# - Revised confidence calculation to prioritize similarity over GPS data
# - GPS weighting now only applied when distance is within the specified threshold
# - Adjusted confidence boost tiers based on similarity score
# Changes in 3.0.20:
# - Fixed montage creation to display all similar images without cropping
# - Adjusted montage height calculation to accommodate all rows of images
# Changes in 3.0.19:
# - Fixed error in create_montage function when handling images without GPS data
# - Improved error handling for missing GPS information
# Changes in 3.0.18:
# - Fixed montage layout to display 6 images per row
# - Corrected issue with blank first image in the montage
# - Adjusted montage to show up to 23 similar images (6 columns * 4 rows - 1 target image)
# Changes in 3.0.17:
# - Modified feature extraction to be more color-agnostic
# - Adjusted similarity threshold handling to potentially include more diverse images
# - Added more detailed logging for image exclusion reasons
# Changes in 3.0.16:
# - Fixed GPS data extraction and display
# - Corrected confidence calculation to properly account for GPS data
# - Added more detailed logging for GPS data extraction
# Changes in 3.0.15:
# - Fixed incorrect confidence calculation when no GPS or object data is available
# - Ensured confidence score does not exceed similarity score without additional data
# Changes in 3.0.14:
# - Fixed issue with scoring data not being visible in the montage
# - Adjusted montage layout to properly display confidence, similarity, object, and GPS scores
# Changes in 3.0.13:
# - Restored scoring data (confidence, similarity, object, GPS) beneath each image in the montage
# - Fixed image rotation issues by respecting EXIF orientation
# Changes in 3.0.12:
# - Fixed similarity threshold enforcement in find_similar_images_cnn function
# - Removed object recognition from similarity and confidence calculations
# - Updated create_montage function to reflect changes in similarity calculation
# Changes in 3.0.11:
# - Reverted and updated create_montage function to match the original layout and information display
# Changes in 3.0.10:
# - Reverted create_montage function to a more traditional layout for consistency with previous versions
# Changes in 3.0.9:
# - Added create_montage function to generate a visual representation of similar images
# Changes in 3.0.8:
# - Improved error handling and logging in find_similar_images_cnn function
# - Added checks for None values in feature extraction and similarity calculation
# - Enhanced debugging output for each image comparison
# Changes in 3.0.7:
# - Fixed issue with undefined target_objects in find_similar_images_cnn function
# - Added more detailed logging for object detection in target image
# Changes in 3.0.6:
# - Added missing extract_gps_info function
# - Ensured all necessary functions are defined and in the correct order
# Changes in 3.0.5:
# - Added more detailed logging throughout the script
# - Implemented a unique run ID for each execution
# - Added a --force-refresh command-line option (functionality to be implemented)
# - Improved error handling and reporting
# - Fixed the detect_objects function to set the confidence threshold correctly
# Changes in 3.0.4:
# - Added error handling for missing dependencies
# - Provided fallback option when YOLO model fails to load
# - Added missing functions: open_image_with_orientation, extract_gps_info, calculate_geo_distance
# - Added fallback detection method using edge detection
# - Modified detect_objects function to use a lower confidence threshold and include the fallback method
# - Updated calculate_object_similarity to handle the new object detection format
# - Added more detailed debug prints in the find_similar_images_cnn function
# Changes in 3.0.3:
# - Added individual contribution values (CNN, Object, GPS) to the montage output for each image
# Changes in 3.0.2:
# - Added model information (MobileNetV2 and YOLOv5) to the montage printout
# Changes from 3.0.1:
# - Added YOLO object detection for semantic understanding
# - Implemented ensemble method with weighted scoring
# - Created composite confidence score combining CNN features, object detection, and GPS data
# - Renamed script to find_image30.py for isolated testing against find_image20.py
# Inherited from find_image20.py (2.0.1):
# - Use of tensorflow.keras.utils.img_to_array
# - Improved error handling in feature extraction
# - Specified input_shape for MobileNetV2

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load YOLO model with error handling
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    use_yolo = True
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Continuing without object detection...")
    use_yolo = False

# Function to open image with correct orientation
def open_image_with_orientation(img_path):
    try:
        img = PilImage.open(img_path)
        
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.ImageIFD.Orientation in exif_dict["0th"]:
                orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
                if orientation == 2:
                    img = img.transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.rotate(180).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img = img.rotate(-90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    
        return img
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

# Function to extract GPS information from image
def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = {}
                        for key in value:
                            decode = ExifTags.GPSTAGS.get(key, key)
                            gps_info[decode] = value[key]
                        
                        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                            lat = gps_info["GPSLatitude"]
                            lon = gps_info["GPSLongitude"]
                            lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                            lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                            if gps_info["GPSLatitudeRef"] == "S":
                                lat = -lat
                            if gps_info["GPSLongitudeRef"] == "W":
                                lon = -lon
                            return (lat, lon)
    except Exception as e:
        print(f"Error extracting GPS info from {img_path}: {e}")
    return None

# Function to calculate geographical distance
def calculate_geo_distance(gps1, gps2):
    if gps1 and gps2:
        return geodesic(gps1, gps2).kilometers
    return None

def fallback_detection(img_path):
    """Fallback method to detect significant structures in an image using edge detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return len(significant_contours) > 0

def detect_objects(img_path):
    """
    Detect objects in an image using YOLO model with a lower confidence threshold.
    Falls back to edge detection if no objects are found.
    """
    print(f"Detecting objects in: {img_path}")
    if not use_yolo:
        return []
    img = cv2.imread(img_path)
    results = yolo_model(img)
    results.conf = 0.25  # Set confidence threshold
    detections = results.pandas().xyxy[0]
    objects = [f"{row['name']}:{row['confidence']:.2f}" for _, row in detections.iterrows()]
    if not objects and fallback_detection(img_path):
        objects = ['unknown_structure:0.50']
    print(f"Detected objects: {objects}")
    return objects

def calculate_object_similarity(objects1, objects2):
    """
    Calculate similarity between two sets of detected objects.
    """
    if not objects1 or not objects2:
        return 0.0
    set1 = set(obj.split(':')[0] for obj in objects1)
    set2 = set(obj.split(':')[0] for obj in objects2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    print(f"Object similarity: {similarity}")
    return similarity

# Function to extract features using MobileNetV2
def extract_features(img_path):
    print(f"Extracting features from: {img_path}")
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.convert('L')  # Convert to grayscale
        img = img.convert('RGB')  # Convert back to RGB (3 channels, but grayscale)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)
        print(f"Feature shape: {features.shape}")
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def calculate_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    print(f"Calculated similarity: {similarity}")
    return similarity

# Updated find_similar_images_cnn function
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
                
                # Only proceed if the similarity meets or exceeds the threshold
                if cnn_similarity >= similarity_threshold:
                    current_objects = detect_objects(file_path)
                    object_similarity = calculate_object_similarity(target_objects, current_objects)
                    current_gps = extract_gps_info(file_path)
                    
                    confidence = cnn_similarity  # Start with the similarity score
                    geo_distance = None

                    if target_gps and current_gps:
                        geo_distance = geodesic(target_gps, current_gps).km
                        if geo_distance <= geo_threshold:
                            if cnn_similarity < 0.48:
                                confidence += 0.3
                            elif 0.48 <= cnn_similarity < 0.6:
                                confidence += 0.2
                            else:
                                confidence += 0.1
                    
                    # Add object similarity to confidence calculation
                    if object_similarity > 0.5:
                        confidence += 0.2
                    elif object_similarity > 0.3:
                        confidence += 0.1
                    
                    confidence = min(confidence, 1.0)
                    
                    print(f"CNN Similarity: {cnn_similarity:.4f}")
                    print(f"Object Similarity: {object_similarity:.4f}")
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
VERSION = "3.0.21"
# Changes in this version:
# - Revised confidence calculation to prioritize similarity over GPS data
# - GPS weighting now only applied when distance is within the specified threshold
# - Adjusted confidence boost tiers based on similarity score
# Changes in 3.0.20:
# - Fixed montage creation to display all similar images without cropping
# - Adjusted montage height calculation to accommodate all rows of images
# Changes in 3.0.19:
# - Fixed error in create_montage function when handling images without GPS data
# - Improved error handling for missing GPS information
# Changes in 3.0.18:
# - Fixed montage layout to display 6 images per row
# - Corrected issue with blank first image in the montage
# - Adjusted montage to show up to 23 similar images (6 columns * 4 rows - 1 target image)
# Changes in 3.0.17:
# - Modified feature extraction to be more color-agnostic
# - Adjusted similarity threshold handling to potentially include more diverse images
# - Added more detailed logging for image exclusion reasons
# Changes in 3.0.16:
# - Fixed GPS data extraction and display
# - Corrected confidence calculation to properly account for GPS data
# - Added more detailed logging for GPS data extraction
# Changes in 3.0.15:
# - Fixed incorrect confidence calculation when no GPS or object data is available
# - Ensured confidence score does not exceed similarity score without additional data
# Changes in 3.0.14:
# - Fixed issue with scoring data not being visible in the montage
# - Adjusted montage layout to properly display confidence, similarity, object, and GPS scores
# Changes in 3.0.13:
# - Restored scoring data (confidence, similarity, object, GPS) beneath each image in the montage
# - Fixed image rotation issues by respecting EXIF orientation
# Changes in 3.0.12:
# - Fixed similarity threshold enforcement in find_similar_images_cnn function
# - Removed object recognition from similarity and confidence calculations
# - Updated create_montage function to reflect changes in similarity calculation
# Changes in 3.0.11:
# - Reverted and updated create_montage function to match the original layout and information display
# Changes in 3.0.10:
# - Reverted create_montage function to a more traditional layout for consistency with previous versions
# Changes in 3.0.9:
# - Added create_montage function to generate a visual representation of similar images
# Changes in 3.0.8:
# - Improved error handling and logging in find_similar_images_cnn function
# - Added checks for None values in feature extraction and similarity calculation
# - Enhanced debugging output for each image comparison
# Changes in 3.0.7:
# - Fixed issue with undefined target_objects in find_similar_images_cnn function
# - Added more detailed logging for object detection in target image
# Changes in 3.0.6:
# - Added missing extract_gps_info function
# - Ensured all necessary functions are defined and in the correct order
# Changes in 3.0.5:
# - Added more detailed logging throughout the script
# - Implemented a unique run ID for each execution
# - Added a --force-refresh command-line option (functionality to be implemented)
# - Improved error handling and reporting
# - Fixed the detect_objects function to set the confidence threshold correctly
# Changes in 3.0.4:
# - Added error handling for missing dependencies
# - Provided fallback option when YOLO model fails to load
# - Added missing functions: open_image_with_orientation, extract_gps_info, calculate_geo_distance
# - Added fallback detection method using edge detection
# - Modified detect_objects function to use a lower confidence threshold and include the fallback method
# - Updated calculate_object_similarity to handle the new object detection format
# - Added more detailed debug prints in the find_similar_images_cnn function
# Changes in 3.0.3:
# - Added individual contribution values (CNN, Object, GPS) to the montage output for each image
# Changes in 3.0.2:
# - Added model information (MobileNetV2 and YOLOv5) to the montage printout
# Changes from 3.0.1:
# - Added YOLO object detection for semantic understanding
# - Implemented ensemble method with weighted scoring
# - Created composite confidence score combining CNN features, object detection, and GPS data
# - Renamed script to find_image30.py for isolated testing against find_image20.py
# Inherited from find_image20.py (2.0.1):
# - Use of tensorflow.keras.utils.img_to_array
# - Improved error handling in feature extraction
# - Specified input_shape for MobileNetV2

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load YOLO model with error handling
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    use_yolo = True
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Continuing without object detection...")
    use_yolo = False

# Function to open image with correct orientation
def open_image_with_orientation(img_path):
    try:
        img = PilImage.open(img_path)
        
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.ImageIFD.Orientation in exif_dict["0th"]:
                orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
                if orientation == 2:
                    img = img.transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.rotate(180).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img = img.rotate(-90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(PilImage.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    
        return img
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None

# Function to extract GPS information from image
def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = {}
                        for key in value:
                            decode = ExifTags.GPSTAGS.get(key, key)
                            gps_info[decode] = value[key]
                        
                        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                            lat = gps_info["GPSLatitude"]
                            lon = gps_info["GPSLongitude"]
                            lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                            lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                            if gps_info["GPSLatitudeRef"] == "S":
                                lat = -lat
                            if gps_info["GPSLongitudeRef"] == "W":
                                lon = -lon
                            return (lat, lon)
    except Exception as e:
        print(f"Error extracting GPS info from {img_path}: {e}")
    return None

# Function to calculate geographical distance
def calculate_geo_distance(gps1, gps2):
    if gps1 and gps2:
        return geodesic(gps1, gps2).kilometers
    return None

def fallback_detection(img_path):
    """Fallback method to detect significant structures in an image using edge detection."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return len(significant_contours) > 0

def detect_objects(img_path):
    """
    Detect objects in an image using YOLO model with a lower confidence threshold.
    Falls back to edge detection if no objects are found.
    """
    print(f"Detecting objects in: {img_path}")
    if not use_yolo:
        return []
    img = cv2.imread(img_path)
    results = yolo_model(img)
    results.conf = 0.25  # Set confidence threshold
    detections = results.pandas().xyxy[0]
    objects = [f"{row['name']}:{row['confidence']:.2f}" for _, row in detections.iterrows()]
    if not objects and fallback_detection(img_path):
        objects = ['unknown_structure:0.50']
    print(f"Detected objects: {objects}")
    return objects

def calculate_object_similarity(objects1, objects2):
    """
    Calculate similarity between two sets of detected objects.
    """
    if not objects1 or not objects2:
        return 0.0
    common_objects = set(obj.split(':')[0] for obj in objects1) & set(obj.split(':')[0] for obj in objects2)
    similarity = len(common_objects) / max(len(objects1), len(objects2))
    print(f"Object similarity: {similarity}")
    return similarity

# Function to extract features using MobileNetV2
def extract_features(img_path):
    print(f"Extracting features from: {img_path}")
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.convert('L')  # Convert to grayscale
        img = img.convert('RGB')  # Convert back to RGB (3 channels, but grayscale)
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)
        print(f"Feature shape: {features.shape}")
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def calculate_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    print(f"Calculated similarity: {similarity}")
    return similarity

# Updated find_similar_images_cnn function
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
                
                # Only proceed if the similarity meets or exceeds the threshold
                if cnn_similarity >= similarity_threshold:
                    current_gps = extract_gps_info(file_path)
                    
                    confidence = cnn_similarity  # Start with the similarity score
                    geo_distance = None

                    if target_gps and current_gps:
                        geo_distance = geodesic(target_gps, current_gps).km
                        if geo_distance <= geo_threshold:
                            if cnn_similarity < 0.48:
                                confidence += 0.3
                            elif 0.48 <= cnn_similarity < 0.6:
                                confidence += 0.2
                            else:
                                confidence += 0.1
                    
                    confidence = min(confidence, 1.0)
                    
                    print(f"CNN Similarity: {cnn_similarity:.4f}")
                    print(f"Confidence: {confidence:.4f}")
                    if geo_distance is not None:
                        print(f"GPS Distance: {geo_distance:.2f} km")
                    else:
                        print("No GPS data available")

                    similar_images.append((file_path, file_name, confidence, cnn_similarity, None, current_gps, geo_distance, None, None, None))
                else:
                    print(f"Similarity {cnn_similarity:.4f} below threshold {similarity_threshold}, skipping")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

    similar_images.sort(key=lambda x: x[2], reverse=True)
    return similar_images

def create_montage(similar_images, target_image_path, similarity_threshold, geo_threshold):
    # Set up montage parameters
    img_width, img_height = 300, 300
    cols = 6
    margin = 10
    font = ImageFont.load_default()

    # Calculate the number of rows needed
    total_images = len(similar_images) + 1  # +1 for the target image
    rows = (total_images + cols - 1) // cols  # Round up division

    # Calculate montage dimensions
    montage_width = cols * (img_width + margin) + margin
    montage_height = rows * (img_height + margin + 60) + margin + 200  # Adjusted for text space and header

    # Create blank montage
    montage = PilImage.new('RGB', (montage_width, montage_height), color='white')
    draw = ImageDraw.Draw(montage)

    # Add header information
    header_text = f"Script: {os.path.basename(sys.argv[0])}\nVersion: {VERSION}\n"
    header_text += f"Image: {os.path.basename(target_image_path)}\nFolder: {os.path.basename(os.path.dirname(similar_images[0][0]))}\n"
    header_text += f"Similarity Threshold: {similarity_threshold}\nGeo Threshold: {geo_threshold} km\n"
    header_text += f"Model: MobileNetV2\n\nConfidence Calculation:\n"
    header_text += "1. Image similarity score\n2. If GPS data available:\n"
    header_text += "   a) similarity < 0.48 and Distance < 0.1km: +0.3 boost\n"
    header_text += "   b) 0.48 <= similarity < 0.6 and Distance < 0.5km: +0.2 boost\n"
    header_text += "   c) Distance < 1km: +0.1 boost\n"
    header_text += "3. Final confidence capped at 1.0"

    draw.text((margin, margin), header_text, fill='black', font=font)

    # Add target image
    target_img = open_image_with_orientation(target_image_path)
    target_img = target_img.convert('RGB')
    target_img.thumbnail((img_width, img_height))
    target_x = margin
    target_y = 200  # Adjust based on header height
    montage.paste(target_img, (target_x, target_y))
    draw.text((target_x, target_y + img_height + 5), f"Conf = 1.0000\nSim = 1.0000\nTarget Image", fill='black', font=font)

    # Add similar images
    for i, (img_path, _, confidence, similarity, _, gps, geo_distance, _, _, _) in enumerate(similar_images, start=1):
        img = open_image_with_orientation(img_path)
        img = img.convert('RGB')
        img.thumbnail((img_width, img_height))
        x = margin + (i % cols) * (img_width + margin)
        y = 200 + (i // cols) * (img_height + margin + 60)  # Adjusted for header and text space
        montage.paste(img, (x, y))
        
        # Add text below the image
        text_y = y + img_height + 5
        draw.text((x, text_y), f"Conf = {confidence:.4f}", fill='black', font=font)
        draw.text((x, text_y + 15), f"Sim = {similarity:.4f}", fill='black', font=font)
        if gps is not None and geo_distance is not None:
            draw.text((x, text_y + 30), f"Dist: {geo_distance:.2f} km", fill='black', font=font)
        else:
            draw.text((x, text_y + 30), "No GPS data", fill='black', font=font)

    # Update the montage save path to use the version number
    version_number = VERSION.replace(".", "")
    montage_path = f'montage{version_number}.jpg'
    montage.save(montage_path)
    print(f"Montage saved as {montage_path}")

def main():
    parser = argparse.ArgumentParser(description='Find similar images')
    parser.add_argument('image_name', help='Name of the target image')
    parser.add_argument('folder_name', help='Name of the folder containing images to compare')
    parser.add_argument('similarity_threshold', type=float, help='Similarity threshold')
    parser.add_argument('geo_threshold', type=float, help='Geographic threshold in km')
    parser.add_argument('--force-refresh', action='store_true', help='Force recalculation of all features')
    args = parser.parse_args()

    print(f"Script name: {sys.argv[0]}")
    print(f"Script version: {VERSION}")
    print(f"Command-line parameters:")
    print(f"  Image name: {args.image_name}")
    print(f"  Folder name: {args.folder_name}")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Geo threshold: {args.geo_threshold} km")
    print(f"  Force refresh: {args.force_refresh}")

    current_dir = os.getcwd()
    target_image_path = os.path.join(current_dir, args.image_name)
    folder_path = os.path.join(current_dir, args.folder_name)

    if not os.path.isfile(target_image_path):
        print(f"Error: Image {args.image_name} not found in the current directory.")
        return
    if not os.path.isdir(folder_path):
        print(f"Error: Folder {args.folder_name} not found in the current directory.")
        return

    similar_images = find_similar_images_cnn(target_image_path, folder_path, args.similarity_threshold, args.geo_threshold)

    if similar_images:
        create_montage(similar_images, target_image_path, args.similarity_threshold, args.geo_threshold)
    else:
        print("No similar images found.")

if __name__ == "__main__":
    main()


