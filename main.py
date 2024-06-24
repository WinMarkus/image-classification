import os
import warnings

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture import MyCNN
from dataset import ImagesDataset


def evaluate_model(model, loader, loss_fn, device):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss += loss_fn(outputs, targets).item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    loss /= len(loader)
    accuracy = correct / total
    model.train()
    return loss, accuracy

def pad_to_square(image):
    _, h, w = image.size()
    max_dim = 100
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    padding = (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h)
    image = transforms.functional.pad(image, padding, fill=0)
    return image

def collate_fn(batch):
    images, targets, *_ = zip(*batch)
    images = [pad_to_square(image) for image in images]
    images = torch.stack(images, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)
    return images, targets

def main(
        results_path,
        learning_rate=1e-3,
        weight_decay=1e-5,
        n_epochs=10,
        device="cuda"
):
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(100),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image_dataset = ImagesDataset(image_dir="training_data")
    
    training_set = torch.utils.data.Subset(
        image_dataset,
        indices=np.arange(int(len(image_dataset) * (3 / 5)))
    )
    validation_set = torch.utils.data.Subset(
        image_dataset,
        indices=np.arange(int(len(image_dataset) * (3 / 5)), int(len(image_dataset) * (4 / 5)))
    )
    test_set = torch.utils.data.Subset(
        image_dataset,
        indices=np.arange(int(len(image_dataset) * (4 / 5)), len(image_dataset))
    )
    
    train_loader = DataLoader(
        training_set,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    net = MyCNN(input_channels=1, 
                hidden_channels=[32, 64, 128], 
                use_batchnormalization=True, 
                num_classes=20, 
                kernel_size=[3, 3, 3], 
                activation_function=torch.nn.ReLU())
    net.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        net.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = evaluate_model(net, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{n_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(results_path, "best_model.pth"))

    print("Finished Training!")
    
    # Load the best model and compute scores on the test set
    net.load_state_dict(torch.load(os.path.join(results_path, "best_model.pth")))
    test_loss, test_accuracy = evaluate_model(net, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Write results to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        rf.write(f"Test Loss: {test_loss:.4f}\n")
        rf.write(f"Test Accuracy: {test_accuracy:.4f}\n")

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
