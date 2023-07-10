import cv2
import os
import imagehash
from PIL import Image

# Function to calculate perceptual hash of an image
def calculate_hash(image_path):
    image = Image.open(image_path).convert('L')  # Open and convert to grayscale
    hash_value = imagehash.average_hash(image)
    return hash_value

# Directory path containing the PNG images
directory = 'screenshots'

# Create a dictionary to store hash values and corresponding file paths
hashes = {}
total = 0
# Loop over PNG images in the directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        image_path = os.path.join(directory, filename)
        image_hash = calculate_hash(image_path)

        # Check if hash value already exists in the dictionary
        if image_hash in hashes.values():
            # Remove the duplicate image
            os.remove(image_path)
            total += 1
            print(f"Removed duplicate image: {filename}")
        else:
            # Add the hash value to the dictionary
            hashes[filename] = image_hash

print(f"Duplicate images removal process completed. {total} images removed.")
