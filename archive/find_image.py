import os
import sys
import numpy as np
from PIL import Image as PilImage, ImageDraw, ImageFont
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Version of the script
VERSION = "1.7.1"

# Load MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess image and extract features using MobileNetV2
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure consistent input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

# Function to find similar images using CNN-based feature matching
def find_similar_images_cnn(target_image_path, folder_path, similarity_threshold=0.7):
    # Extract features for the target image
    target_features = extract_features(target_image_path)

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

                # Debug: Print the similarity score for the current image
                print(f"Checking {file_name}: Similarity = {similarity:.4f}")

                # If similarity is above the threshold, consider it similar
                if similarity >= similarity_threshold:
                    similar_images.append((file_path, file_name, similarity))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Sort images by similarity in descending order
    similar_images.sort(key=lambda x: x[2], reverse=True)

    return similar_images

# Function to create a montage of similar images with similarity scores
def create_montage(similar_images, target_image_path, script_info, output_file="montage.png"):
    # Max size of each image in the montage (increased size)
    image_size = (200, 200)
    images_per_row = 3

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
    target_image = PilImage.open(target_image_path).resize(image_size)
    montage_image.paste(target_image, (0, 100))

    # Add a white block for the similarity score (which is 1.0 for the search image)
    draw.rectangle([(0, 100 + image_size[1]), (image_size[0], 100 + image_size[1] + 40)], fill=(255, 255, 255))
    draw.text((10, 100 + image_size[1] + 10), f"Similarity = 1.0000", fill=(0, 0, 0), font=font)

    # Start placing matched images in rows of 3
    for idx, (img_path, _, similarity) in enumerate(similar_images):
        img = PilImage.open(img_path).resize(image_size)
        col = idx % images_per_row  # Use modulo to determine column position
        row = (idx // images_per_row) + 1  # First row is taken by the target image

        x_offset = col * image_size[0]
        y_offset = row * (image_size[1] + 40) + 100

        # Paste image
        montage_image.paste(img, (x_offset, y_offset))

        # Draw the white block and similarity score below each image
        draw.rectangle([(x_offset, y_offset + image_size[1]), 
                        (x_offset + image_size[0], y_offset + image_size[1] + 40)], fill=(255, 255, 255))
        draw.text((x_offset + 10, y_offset + image_size[1] + 10), f"Similarity = {similarity:.4f}", fill=(0, 0, 0), font=font)

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
    if len(sys.argv) < 4:
        print("Usage: python find_image_cnn.py <image_name> <folder_name> <similarity_threshold>")
        sys.exit(1)

    # Get the image name, folder name, and similarity threshold from command-line arguments
    image_name = sys.argv[1]
    folder_name = sys.argv[2]
    similarity_threshold = float(sys.argv[3])  # Convert similarity threshold to a float

    # Print the command-line parameters
    print(f"Command-line parameters:")
    print(f"  Image name: {image_name}")
    print(f"  Folder name: {folder_name}")
    print(f"  Similarity threshold: {similarity_threshold}")

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
    similar_images = find_similar_images_cnn(target_image_path, folder_path, similarity_threshold)

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


