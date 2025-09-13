import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
num_classes = 10
learning_rate = 1e-4
batch_size = 64
num_epochs = 1000
alpha = 1  # Weighting factor for the reconstruction loss

# Load and preprocess MNIST dataset
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a shared weight parameter
shared_weight = nn.Parameter(torch.randn(input_size, num_classes) * 0.01)

# --- MODEL DEFINITIONS ---

# 1. Encoder (Classifier)
class Encoder(nn.Module):
    def __init__(self, weight):
        super(Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.weight = weight
    
    def forward(self, x):
        x = self.flatten(x)
        x = x @ self.weight
        return x

# 2. Decoder (transpose weight)
class Decoder(nn.Module):
    def __init__(self, weight):
        super(Decoder, self).__init__()
        self.weight = weight
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x @ self.weight.T
        return self.sigmoid(x)

# 3. DecoderInv (pseudo inverting weight) 
class DecoderInv(nn.Module):
    def __init__(self, weight):
        super(DecoderInv, self).__init__()
        self.weight = weight
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        weight = torch.linalg.pinv(self.weight)
        x = x @ weight
        return x


# Initialize models with the shared weight parameter
model_encoder = Encoder(shared_weight)
model_mean = Decoder(shared_weight)

model_encoder.compile()
model_mean.compile()


# --- LOSS FUNCTION DEFINITIONS ---

criterion_cls = nn.CrossEntropyLoss()
criterion_recon = nn.L1Loss()

# Use a single optimizer for the shared weight
optimizer = optim.Adam([shared_weight], lr=learning_rate)

# --- TRAINING & TESTING ---

def train():
    loss_history = []
    best_loss = float('inf')
    
    model_encoder.train()
    model_mean.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_cls_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            flat_data = torch.flatten(data, start_dim=1)
          
            # --- Forward Pass ---
            # 1. Classification path
            output = model_encoder(data)
            
            # 2. Reconstruction path (Mean and Variance)
            reconstructed_mean = model_mean(output)

            # --- Loss Calculation ---
            loss_cls = criterion_cls(output, target)
            
            # reconstruction loss
            loss_recon = criterion_recon(reconstructed_mean, flat_data)

            # Combined loss
            loss = loss_cls + alpha * loss_recon
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cls_loss += loss_cls.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)

        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Combined Loss: {avg_loss:.4f}, Reconstruction Loss: {avg_recon_loss*alpha:.4f}, Classification Loss: {avg_cls_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(shared_weight, 'shared_weight.pth')

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Loss vs. Epoch')
    plt.grid(True)
    plt.show()

def test():
    
    # model_encoder.load_state_dict({'weight': shared_weight_loaded}, strict=False)
    # model_mean.load_state_dict({'weight': shared_weight_loaded}, strict=False)
    # model_log_var.load_state_dict({'weight': shared_weight_loaded}, strict=False)
    
    model_encoder.eval()
    model_mean.eval()

    correct = 0
    total = 0
    test_loss_cls = 0
    test_loss_recon = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            flat_data = torch.flatten(data, start_dim=1)
            
            # Classification
            output = model_encoder(data)
            test_loss_cls += criterion_cls(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Reconstruction
            reconstructed_mean = model_mean(output)
            test_loss_recon += criterion_recon(reconstructed_mean, flat_data).item()

    test_loss_cls /= len(test_loader)
    test_loss_recon /= len(test_loader)
    
    accuracy = 100. * correct / total
    print(f'\nTest set: Classification Loss: {test_loss_cls:.4f}, Reconstruction Loss: {test_loss_recon:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')


def generate_and_plot_images(weight):
    model_mean.load_state_dict({'weight': weight}, strict=False)    
    model_mean.eval()
    
    print("\nGenerating images from one-hot vectors...")
    one_hot_vectors = torch.eye(num_classes)
    
    with torch.no_grad():
        generated_means = model_mean(one_hot_vectors)

    generated_means = generated_means.reshape(num_classes, 28, 28)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        # Plot generated mean
        ax.imshow(generated_means[i].cpu().numpy(), cmap='gray')
        ax.set_title(f"Generated: {i}")
        ax.axis('off')

    plt.suptitle("Generated Digit Mean Images from One-Hot Vectors", fontsize=14)
    plt.tight_layout()
    plt.show()


# Run training, testing, and generation
if __name__ == '__main__':
    train()
    test()
    generate_and_plot_images(weight=torch.load('shared_weight.pth'))