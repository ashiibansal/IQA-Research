import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import the classes from your other files
from dataset import DegradationDataset
from model import RestorationParameterPredictor

# --- 1. Setup Transformations ---
resnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# --- 2. Instantiate the Dataset & DataLoader ---
my_dataset = DegradationDataset(
    csv_file='degradation_labels.csv', 
    root_dir='degraded_images', 
    transform=resnet_transforms
)

train_loader = DataLoader(dataset=my_dataset, batch_size=32, shuffle=True)

# --- 3. THE TRAINING LOOP ---
def train_model():
    # Detect if you have a GPU available (makes training 10x faster)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load the model and send it to the GPU/CPU
    model = RestorationParameterPredictor().to(device)

    # Define Loss Function: Mean Squared Error (perfect for predicting numbers)
    criterion = nn.MSELoss() 
    
    # Define Optimizer: Adam updates the model's weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Epochs: How many times the model will look at the ENTIRE dataset
    num_epochs = 10 

    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train() # Set the model to "learning mode"
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"=== Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f} ===")

    # --- 4. SAVE THE MODEL ---
    torch.save(model.state_dict(), "restoration_model.pth")
    print("Training complete! Model saved as 'restoration_model.pth'")

if __name__ == "__main__":
    train_model()