import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Import your custom classes
from dataset import DegradationDataset
from model import RestorationParameterPredictor

bounds = {
    'blur': (0.0, 3.0),
    'noise': (0.0, 25.0),
    'jpeg': (10.0, 100.0),
    'gamma': (0.5, 1.5)
}

def denormalize(value, param_name):
    min_val, max_val = bounds[param_name]
    return value * (max_val - min_val) + min_val

resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = DegradationDataset(
    csv_file='degradation_labels.csv', 
    root_dir='degraded_images',        
    transform=resnet_transforms
)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestorationParameterPredictor().to(device)
    model.load_state_dict(torch.load("restoration_model.pth", map_location=device, weights_only=True))
    model.eval() 

    total_mae_blur, total_mae_noise, total_mae_jpeg, total_mae_gamma = 0.0, 0.0, 0.0, 0.0
    num_samples = 0

    print("Starting Evaluation on Test Data...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            # --- THE FIXED MATH ---
            pred_blur = denormalize(predictions[0][0].item(), 'blur')
            pred_noise = denormalize(predictions[0][1].item(), 'noise')
            pred_jpeg = denormalize(predictions[0][2].item(), 'jpeg')
            pred_gamma = denormalize(predictions[0][3].item(), 'gamma')

            true_blur = denormalize(labels[0][0].item(), 'blur')
            true_noise = denormalize(labels[0][1].item(), 'noise')
            true_jpeg = denormalize(labels[0][2].item(), 'jpeg')
            true_gamma = denormalize(labels[0][3].item(), 'gamma')

            total_mae_blur += abs(true_blur - pred_blur)
            total_mae_noise += abs(true_noise - pred_noise)
            total_mae_jpeg += abs(true_jpeg - pred_jpeg)
            total_mae_gamma += abs(true_gamma - pred_gamma)
            
            num_samples += 1

    print("\n=== FINAL EVALUATION RESULTS (Mean Absolute Error) ===")
    print(f"Total Test Images: {num_samples}")
    print(f"Blur MAE:   Off by an average of {total_mae_blur / num_samples:.3f} pixels")
    print(f"Noise MAE:  Off by an average of {total_mae_noise / num_samples:.3f} variance")
    print(f"JPEG MAE:   Off by an average of {total_mae_jpeg / num_samples:.2f} quality points")
    print(f"Gamma MAE:  Off by an average of {total_mae_gamma / num_samples:.3f} exposure")

if __name__ == "__main__":
    evaluate_model()