

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

IMAGE_SIZE = 176 
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4 
DATA_DIR = "dataset"  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc.item())
        
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.4f}')
        
        print()
    
    torch.save(model.state_dict(), 'final_model.pth')
    
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

def plot_training_history(train_loss, train_acc, val_loss, val_acc):
   
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def convert_pytorch_to_keras(model_path, output_path):
   
    try:
        import onnx
        from onnx2keras import onnx_to_keras
        
        model = AlzheimerCNN(NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        torch.onnx.export(model, dummy_input, "temp_model.onnx", 
                         verbose=False, input_names=['input'], 
                         output_names=['output'])
        
        onnx_model = onnx.load("temp_model.onnx")
        
        k_model = onnx_to_keras(onnx_model, ['input'])
        
        k_model.save(output_path)
        print(f"Model successfully converted and saved to {output_path}")
        
        os.remove("temp_model.onnx")
        
    except ImportError:
        print("Error: Required libraries for conversion not found.")
        print("To convert model, please install: pip install onnx onnx2keras")
        print(f"Model saved in PyTorch format at {model_path}")
        print("You will need to adjust your app.py to load a PyTorch model instead")
    
def main():
    print("Starting Alzheimer's Disease Detection Training with PyTorch...")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please download the dataset from Kaggle: https://www.kaggle.com/datasets/ninadaithal/imagesoasis")
        print("And extract it to create the following structure:")
        print(f"{DATA_DIR}/")
        print("├── Mild_Demented/")
        print("├── Moderate_Demented/")
        print("├── Non_Demented/")
        print("└── Very_Mild_Demented/")
        return
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print("Loading and preprocessing data...")

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms['train'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = data_transforms['val']
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    model = AlzheimerCNN(NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training model...")
    model, train_loss, train_acc, val_loss, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS
    )
    
    plot_training_history(train_loss, train_acc, val_loss, val_acc)
    
    with open('Accuracy.txt', 'w') as f:
        f.write(f"Final Training Accuracy: {train_acc[-1]*100:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_acc[-1]*100:.2f}%\n")
    
    print("Training completed successfully!")
    print(f"Best model saved as 'best_model.pth'")
    print(f"Final model saved as 'final_model.pth'")
    
    try:
        print("\nAttempting to convert model to Keras format for use with your app.py...")
        convert_pytorch_to_keras('best_model.pth', 'cnn_model.h5')
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        print("You'll need to either:")
        print("1. Install the conversion libraries (onnx, onnx2keras)")
        print("2. Or modify your app.py to use the PyTorch model directly")
        print("3. Or fix your TensorFlow installation to use the original script")

if __name__ == "__main__":
    main()