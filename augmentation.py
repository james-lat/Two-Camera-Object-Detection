import cv2
import numpy as np
import random
import os
import time

# --- Configuration ---
INPUT_FOLDER = 'Laboro_tomato/val/images'
OUTPUT_FOLDER = 'Laboro_tomato/val/images_aug'
TARGET_SIZE = (640, 640)
AUGMENTATION_CHANCE = 0.5 

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- Augmentation Functions ---
# NOTE: Removed resizing from here to improve performance. 
# We now assume the image passed in is already resized.

def vertical_flip(image):
    return cv2.flip(image, 0)

def horizontal_flip(image):
    return cv2.flip(image, 1)

def mirror_flip(image):
    return cv2.flip(image, -1)

def brightness_change(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    value = random.randint(-50, 50) 
    
    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = 0 - value
        v[v < lim] = 0
        v[v >= lim] -= abs(value)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# --- Main Processing Pipeline ---

def process_dataset():
    # Ensure input and output are not the same to prevent infinite loops
    if os.path.abspath(INPUT_FOLDER) == os.path.abspath(OUTPUT_FOLDER):
        print("Error: Input and Output folders must be different!")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(files)
    print(f"Found {total_files} images. Starting processing...")
    
    count_augmented = 0
    start_time = time.time()

    # Enumerate allows us to get the index (i) for the progress counter
    for i, filename in enumerate(files):
        img_path = os.path.join(INPUT_FOLDER, filename)
        original_img = cv2.imread(img_path)

        if original_img is None:
            continue

        base_name = os.path.splitext(filename)[0]

        # OPTIMIZATION: Resize ONCE here, use this variable for everything else
        resized_img = cv2.resize(original_img, TARGET_SIZE)
        
        # Save the resized original
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_orig.jpg"), resized_img)

        # Augmentation Logic
        if random.random() < AUGMENTATION_CHANCE:
            count_augmented += 1
            
            augmentations = [
                (vertical_flip, "_vert"),
                (horizontal_flip, "_hori"),
                (mirror_flip, "_mirror"),
                (brightness_change, "_bright")
            ]

            for func, suffix in augmentations:
                try:
                    # Pass the ALREADY RESIZED image
                    aug_img = func(resized_img)
                    new_filename = f"{base_name}{suffix}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, new_filename), aug_img)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # PROGRESS FEEDBACK
        # Print status every 10 images so you know it's not frozen
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_files} images...", end='\r')

    end_time = time.time()
    print(f"\nDone! Processed {total_files} images in {end_time - start_time:.2f} seconds.")
    print(f"Number of images selected for augmentation: {count_augmented}")

if __name__ == "__main__":
    process_dataset()
