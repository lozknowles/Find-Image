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

# Version of the script
VERSION = "2.0.1"
# Changes in this version:
# - Updated to use tensorflow.keras.utils.img_to_array instead of deprecated function
# - Improved error handling in feature extraction
# - Specified input_shape for MobileNetV2 to suppress warnings and ensure correct weight loading
# Changes from 2.0.0:
# - Added L2 normalization to CNN feature extraction
# - Updated to use sklearn's normalize function
# Changes from 1.x:
# - Significantly increased vertical spacing between images
# - Ensured text is not obscured by adjusting image placement
# - Reorganized layout to prevent text overlap

# Load MobileNetV2 model pre-trained on ImageNet with specified input shape
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Function to open image respecting EXIF orientation
def open_image_with_orientation(img_path):
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

# Function to preprocess image and extract features using MobileNetV2
def extract_features(img_path):
    try:
        img = open_image_with_orientation(img_path)
        if img is None:
            return None
        img = img.resize((224, 224))  # Ensure consistent input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = normalize(features)  # L2 normalization
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def extract_gps_info(img_path):
    try:
        with PilImage.open(img_path) as img:
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in ExifTags.TAGS
            }
            if 'GPSInfo' in exif:
                gps_info = exif['GPSInfo']
                lat = gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600
                lon = gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600
                if gps_info[1] == 'S': lat = -lat
                if gps_info[3] == 'W': lon = -lon
                return (lat, lon)
    except:
        pass
    return None

def calculate_geo_distance(coord1, coord2):
    if coord1 and coord2:
        return geodesic(coord1, coord2).kilometers
    return None

# Function to find similar images using CNN-based feature matching
def find_similar_images_cnn(target_image_path, folder_path, similarity_threshold=0.7, geo_threshold=10):
    # Extract features for the target image
    target_features = extract_features(target_image_path)
    target_gps = extract_gps_info(target_image_path)
    
    if target_gps:
        print(f"Search image GPS: {target_gps}")
    else:
        print("Search image has no GPS data")

    similar_images = []

    # Iterate through all images in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Extract features for the current image
                current_features = extract_features(file_path)

                # Compute cosine similarity between the target and current image
                similarity = cosine_similarity(target_features, current_features)[0][0]

                current_gps = extract_gps_info(file_path)
                geo_distance = calculate_geo_distance(target_gps, current_gps) if target_gps and current_gps else None

                # Base confidence is the similarity score
                confidence = similarity

                if geo_distance is not None:
                    if similarity > 0.48 and geo_distance < 0.1:
                        # Significant boost for high similarity and very close distance
                        confidence += 0.3
                    elif geo_distance <= 0.5:
                        # Small boost for close images
                        confidence += 0.05
                    elif geo_distance > 1.0:
                        # Small penalty for distant images
                        confidence -= 0.05
                else:
                    # Small boost for high similarity images without GPS data
                    if similarity > 0.48:
                        confidence += 0.1

                # Ensure confidence doesn't exceed 1.0
                confidence = min(confidence, 1.0)

                print(f"Checking {file_name}: Similarity = {similarity:.4f}, Confidence = {confidence:.4f}, GPS = {current_gps}, Distance = {geo_distance:.2f}km" if geo_distance else f"Checking {file_name}: Similarity = {similarity:.4f}, Confidence = {confidence:.4f}, GPS = {current_gps}")

                if confidence >= similarity_threshold:
                    similar_images.append((file_path, file_name, confidence, similarity, current_gps, geo_distance))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Sort primarily by confidence, then by similarity
    similar_images.sort(key=lambda x: (x[2], x[3]), reverse=True)

    return similar_images

# Function to create a montage of similar images with similarity scores
def create_montage(similar_images, target_image_path, script_info, output_file="montage.png"):
    image_size = (200, 200)
    images_per_row = 6
    info_height = 250  # Height for information at the top
    image_spacing = 10  # Horizontal spacing between images
    vertical_spacing = 100  # Significantly increased vertical spacing

    num_images = len(similar_images) + 1
    num_rows = (num_images // images_per_row) + (1 if num_images % images_per_row != 0 else 0)

    montage_width = images_per_row * (image_size[0] + image_spacing) - image_spacing
    montage_height = info_height + num_rows * (image_size[1] + vertical_spacing)
    montage_image = PilImage.new('RGB', (montage_width, montage_height), (255, 255, 255))

    draw = ImageDraw.Draw(montage_image)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # Draw script information at the top
    y_offset = 10
    for line in script_info.split('\n'):
        draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += 15

    # Handle target image
    target_image = open_image_with_orientation(target_image_path).resize(image_size)
    montage_image.paste(target_image, (0, info_height))

    draw.text((10, info_height + image_size[1] + 5), f"Conf = 1.0000", fill=(0, 0, 0), font=font)
    draw.text((10, info_height + image_size[1] + 25), f"Sim = 1.0000", fill=(0, 0, 0), font=font)
    draw.text((10, info_height + image_size[1] + 45), "Target Image", fill=(0, 0, 0), font=font)

    for idx, (img_path, _, confidence, similarity, gps, distance) in enumerate(similar_images):
        img = open_image_with_orientation(img_path).resize(image_size)
        col = idx % images_per_row
        row = (idx // images_per_row) + 1  # +1 to account for the target image row

        x_offset = col * (image_size[0] + image_spacing)
        y_offset = info_height + row * (image_size[1] + vertical_spacing)

        montage_image.paste(img, (x_offset, y_offset))

        # Draw text below each image
        text_y = y_offset + image_size[1] + 5
        draw.text((x_offset + 10, text_y), f"Conf = {confidence:.4f}", fill=(0, 0, 0), font=font)
        draw.text((x_offset + 10, text_y + 20), f"Sim = {similarity:.4f}", fill=(0, 0, 0), font=font)
        
        if distance is not None:
            distance_text = f"Dist: {distance:.2f} km"
        else:
            distance_text = "No GPS data"
        draw.text((x_offset + 10, text_y + 40), distance_text, fill=(0, 0, 0), font=font)

    montage_image.save(output_file)
    print(f"Montage saved as {output_file}")

# Main function to handle command-line input
def main():
    # Script information
    script_name = sys.argv[0]
    print(f"Script name: {script_name}")
    print(f"Script version: {VERSION}")

    # Check if enough arguments are passed
    if len(sys.argv) < 5:
        print("Usage: python find_image_cnn.py <image_name> <folder_name> <similarity_threshold> <geo_threshold>")
        sys.exit(1)

    # Get the image name, folder name, similarity threshold, and geo threshold from command-line arguments
    image_name = sys.argv[1]
    folder_name = sys.argv[2]
    similarity_threshold = float(sys.argv[3])  # Convert similarity threshold to a float
    geo_threshold = float(sys.argv[4])  # Convert geo threshold to a float

    # Print the command-line parameters
    print(f"Command-line parameters:")
    print(f"  Image name: {image_name}")
    print(f"  Folder name: {folder_name}")
    print(f"  Similarity threshold: {similarity_threshold}")
    print(f"  Geo threshold: {geo_threshold} km")

    # Construct paths (assuming folder and image are in the same directory as the script)
    current_dir = os.getcwd()
    target_image_path = os.path.join(current_dir, image_name)
    folder_path = os.path.join(current_dir, folder_name)

    # Check if the image and folder exist
    if not os.path.isfile(target_image_path):
        print(f"Error: Image {image_name} not found in the current directory.")
        return
    if not os.path.isdir(folder_path):
        print(f"Error: Folder {folder_name} not found in the current directory.")
        return

    # Call the function to find similar images
    similar_images = find_similar_images_cnn(target_image_path, folder_path, similarity_threshold, geo_threshold)

    # Prepare script info for the top of the montage
    script_info = f"Script: {script_name}\nVersion: {VERSION}\nImage: {image_name}\nFolder: {folder_name}\n"
    script_info += f"Similarity Threshold: {similarity_threshold}\nGeo Threshold: {geo_threshold} km\n"
    script_info += f"Model: MobileNetV2\n\n"
    script_info += "Confidence Calculation:\n"
    script_info += "1. Base confidence = Similarity score\n"
    script_info += "2. If GPS data available:\n"
    script_info += "   - If Similarity > 0.48 and Distance < 0.1km: +0.3 boost\n"
    script_info += "   - If Distance <= 0.5km: +0.05 boost\n"
    script_info += "   - If Distance > 1.0km: -0.05 penalty\n"
    script_info += "3. If no GPS data and Similarity > 0.48: +0.1 boost\n"
    script_info += "4. Final confidence capped at 1.0"

    # If similar images are found, create a montage
    if similar_images:
        create_montage(similar_images, target_image_path, script_info)
    else:
        print("No similar images found.")

# Run the main function
if __name__ == "__main__":
    main()


