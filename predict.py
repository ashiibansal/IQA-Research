import torch
from torchvision import transforms
from PIL import Image
from model import RestorationParameterPredictor

# --- 1. Settings ---
MODEL_PATH = "restoration_model.pth"
# Updated to match your .jpeg file
IMAGE_PATH = "boat.png" 

# Use the same bounds from our dataset to de-normalize the results
bounds = {
    'blur': (0.0, 3.0),
    'noise': (0.0, 25.0),
    'jpeg': (10.0, 100.0),
    'gamma': (0.5, 1.5)
}

def denormalize(value, param_name):
    min_val, max_val = bounds[param_name]
    return value * (max_val - min_val) + min_val

# --- 2. Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RestorationParameterPredictor().to(device)
# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- 3. Preprocess Image ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_degradation(img_path):
    # Load and convert image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        preds = model(img_tensor).squeeze(0)  # Remove batch dimension

    # Now preds is shape (4,)
    blur_pred = denormalize(preds[0].item(), 'blur')
    noise_pred = denormalize(preds[1].item(), 'noise')
    jpeg_pred = denormalize(preds[2].item(), 'jpeg')
    gamma_pred = denormalize(preds[3].item(), 'gamma')

    results = {
        "Blur Sigma": blur_pred,
        "Noise Level": noise_pred,
        "JPEG Quality": jpeg_pred,
        "Gamma Shift": gamma_pred
    }

    return results

if __name__ == "__main__":
    try:
        print(f"Analyzing: {IMAGE_PATH}...")
        metrics = predict_degradation(IMAGE_PATH)
        print("\n--- Predicted Degradation Metrics ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    except FileNotFoundError:
        print(f"Error: Could not find image at {IMAGE_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")