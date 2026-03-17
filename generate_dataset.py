import cv2
import numpy as np
import os
import random
import csv
import glob

# --- Configuration ---
INPUT_DIR = "clean_images"
OUTPUT_DIR = "degraded_images"
CSV_FILE = "degradation_labels.csv"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_degradations(img):
    """
    Applies random levels of Exposure, Blur, Noise, and JPEG compression.
    Returns the ruined image and the exact parameters used.
    """
    
    # 1. Exposure (Gamma Correction)
    # Range: 0.5 (bright/washed out) to 1.5 (dark/underexposed). 1.0 is no change.
    gamma = round(random.uniform(0.5, 1.5), 2)
    inv_gamma = 1.0 / gamma
    # Build a Look-Up Table (LUT) to apply the math quickly
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 2. Gaussian Blur
    # Range: 0.0 (no blur) to 3.0 (heavy blur)
    blur_sigma = round(random.uniform(0.0, 3.0), 2)
    if blur_sigma > 0:
        # We use (0,0) so the kernel size is automatically computed from the sigma value
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)

    # 3. Gaussian Noise
    # Range: 0 (no noise) to 25 (heavy TV static)
    noise_std = round(random.uniform(0, 25), 2)
    if noise_std > 0:
        # Generate random noise map
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        # Add noise to image, and use np.clip to ensure pixels stay between 0 and 255
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. JPEG Compression Quality
    # Range: 10 (terrible blocky quality) to 100 (perfect quality)
    jpeg_quality = random.randint(10, 100)

    return img, blur_sigma, noise_std, jpeg_quality, gamma

def main():
    # Grab all .jpg and .png images from the input folder
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    
    if not image_paths:
        print(f"Error: No images found in {INPUT_DIR}. Please add some images!")
        return

    # Open the CSV file to start logging our "recipes"
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['image_name', 'blur_sigma', 'noise_std', 'jpeg_quality', 'gamma_exposure'])

        print(f"Starting degradation process for {len(image_paths)} images...")

        for count, img_path in enumerate(image_paths):
            # Read the clean image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Apply our random degradations
            ruined_img, blur_sigma, noise_std, jpeg_quality, gamma = apply_degradations(img)

            # Create a new filename (e.g., degraded_0001.jpg)
            filename = f"degraded_{count:04d}.jpg"
            output_path = os.path.join(OUTPUT_DIR, filename)

            # Save the image WITH the specific JPEG compression quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            cv2.imwrite(output_path, ruined_img, encode_param)

            # Log the exact parameters to the CSV
            writer.writerow([filename, blur_sigma, noise_std, jpeg_quality, gamma])

            # Print progress every 10 images
            if count % 10 == 0:
                print(f"Processed {count}/{len(image_paths)} images...")

    print(f"Done! Dataset generated in '{OUTPUT_DIR}' and labels saved to '{CSV_FILE}'.")

if __name__ == "__main__":
    main()