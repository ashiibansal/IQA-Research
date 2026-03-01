import cv2
import numpy as np

def restore_image(image_path, blur_sigma, noise_level, gamma):
    img = cv2.imread(image_path)
    
    # 1. Reverse Gamma (Exposure)
    # If model predicted 0.84, we apply 1/0.84 to neutralize it
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    
    # 2. Denoising
    # We use the predicted Noise Level to set the filter strength 'h'
    if noise_level > 5:
        img = cv2.fastNlMeansDenoisingColored(img, None, h=noise_level, hColor=10, templateWindowSize=7, searchWindowSize=21)
        
    # 3. Sharpening (Simple Inverse for Blur)
    if blur_sigma > 0.5:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
    cv2.imwrite("restored_output.jpg", img)
    print("Restoration complete! Saved as 'restored_output.jpg'")

# Use your actual model outputs here
restore_image("test_image.jpeg", blur_sigma=1.0893, noise_level=16.2089, gamma=0.8429)