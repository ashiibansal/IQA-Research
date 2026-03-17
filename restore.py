import cv2
import numpy as np
import sys


def restore_image(image_path, blur_sigma, noise_level, gamma):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    # ------------------------------------------------
    # 1. STRONG DENOISING
    # ------------------------------------------------
    h_strength = min(noise_level / 10, 15)

    img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=h_strength,
        hColor=h_strength,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # ------------------------------------------------
    # 2. JPEG ARTIFACT REDUCTION
    # ------------------------------------------------
    img = cv2.bilateralFilter(
        img,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    # ------------------------------------------------
    # 3. DEBLURRING (Unsharp Mask)
    # ------------------------------------------------
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)

    img = cv2.addWeighted(
        img,
        1.7,
        gaussian,
        -0.7,
        0
    )

    # ------------------------------------------------
    # 4. CONTRAST ENHANCEMENT (CLAHE)
    # ------------------------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------
    # 5. CONTROLLED GAMMA CORRECTION
    # ------------------------------------------------
    gamma_correct = max(0.85, min(1.15, 1 / gamma))

    table = np.array([
        ((i / 255.0) ** gamma_correct) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    img = cv2.LUT(img, table)

    # ------------------------------------------------
    # SAVE RESULT
    # ------------------------------------------------
    output_path = "restored_image.jpg"
    cv2.imwrite(output_path, img)

    print(f"Restored image saved as: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    else:
        IMAGE_PATH = "degraded_images/degraded_0001.jpg"

    # predicted values from your model
    blur_sigma = 3.083
    noise_level = 25.4089
    gamma = 0.73

    restore_image(IMAGE_PATH, blur_sigma, noise_level, gamma)