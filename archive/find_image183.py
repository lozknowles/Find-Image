import os
import sys
import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFont, ExifTags
import piexif
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

# Version of the script
VERSION = "1.8.3"
# Changes in this version:
# - Display distance in km instead of raw GPS coordinates when available
# Changes from previous version:
# - Increased the number of images per row in the montage from 3 to 6
# - Added support for correct image orientation using EXIF data
# - Implemented GPS data extraction and distance calculation
# - Updated confidence calculation to include both visual similarity and geo-proximity

# Load MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

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
    img = open_image_with_orientation(img_path)
    img = img.resize((224, 224))  # Ensure consistent input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

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

                print(f"Checking {file_name}: Similarity = {similarity:.4f}, GPS = {current_gps}, Distance = {geo_distance:.2f}km" if geo_distance else f"Checking {file_name}: Similarity = {similarity:.4f}, GPS = {current_gps}")

                confidence = similarity
                if geo_distance is not None and geo_distance <= geo_threshold:
                    confidence += (1 - (geo_distance / geo_threshold)) * 0.2  # Boost confidence by up to 0.2 based on proximity

                # If similarity is above the threshold, consider it similar
                if confidence >= similarity_threshold:
                    similar_images.append((file_path, file_name, confidence, current_gps, geo_distance))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Sort images by similarity in descending order
    similar_images.sort(key=lambda x: x[2], reverse=True)

    return similar_images

# Function to create a montage of similar images with similarity scores
def create_montage(similar_images, target_image_path, script_info, output_file="montage.png"):
    # Max size of each image in the montage (increased size)
    image_size = (200, 200)
    images_per_row = 6  # Changed from 3 to 6

    # Calculate number of rows and columns needed
    num_images = len(similar_images) + 1  # Include the search image
    num_rows = (num_images // images_per_row) + (1 if num_images % images_per_row != 0 else 0)

    # Create a blank canvas for the montage with space for the white block beneath each image
    montage_width = images_per_row * image_size[0]
    montage_height = num_rows * (image_size[1] + 40) + 100  # 40px for the white block and 100px for header
    montage_image = PilImage.new('RGB', (montage_width, montage_height), (255, 255, 255))

    # Add script information at the top
    draw = ImageDraw.Draw(montage_image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), script_info, fill=(0, 0, 0), font=font)

    # Paste the target image at the top-left corner of the montage
    target_image = open_image_with_orientation(target_image_path).resize(image_size)
    montage_image.paste(target_image, (0, 100))

    # Add a white block for the similarity score (which is 1.0 for the search image)
    draw.rectangle([(0, 100 + image_size[1]), (image_size[0], 100 + image_size[1] + 40)], fill=(255, 255, 255))
    draw.text((10, 100 + image_size[1] + 10), f"Similarity = 1.0000", fill=(0, 0, 0), font=font)

    target_gps = extract_gps_info(target_image_path)
    gps_text = "No GPS data" if not target_gps else "Search image GPS"
    draw.text((10, 100 + image_size[1] + 25), gps_text, fill=(0, 0, 0), font=font)

    # Start placing matched images in rows of 6
    for idx, (img_path, _, confidence, gps, distance) in enumerate(similar_images):
        img = open_image_with_orientation(img_path).resize(image_size)
        col = (idx + 1) % images_per_row  # +1 because the first slot is taken by the target image
        row = ((idx + 1) // images_per_row) + 1  # +1 for the same reason as above

        x_offset = col * image_size[0]
        y_offset = row * (image_size[1] + 40) + 100

        # Paste image
        montage_image.paste(img, (x_offset, y_offset))

        # Draw the white block and similarity score below each image
        draw.rectangle([(x_offset, y_offset + image_size[1]), 
                        (x_offset + image_size[0], y_offset + image_size[1] + 40)], fill=(255, 255, 255))
        draw.text((x_offset + 10, y_offset + image_size[1] + 10), f"Similarity = {confidence:.4f}", fill=(0, 0, 0), font=font)

        if distance is not None:
            distance_text = f"Distance: {distance:.2f} km"
        else:
            distance_text = "No GPS data"
        draw.text((x_offset + 10, y_offset + image_size[1] + 25), distance_text, fill=(0, 0, 0), font=font)

    # Save the montage image
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
    script_info = f"Script: {script_name}\nVersion: {VERSION}\nImage: {image_name}\nFolder: {folder_name}\nModel: MobileNetV2"

    # If similar images are found, create a montage
    if similar_images:
        create_montage(similar_images, target_image_path, script_info)
    else:
        print("No similar images found.")

# Run the main function
if __name__ == "__main__":
    main()


