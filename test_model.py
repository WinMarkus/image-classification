import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from architecture import MyCNN
from dataset import ImagesDataset


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(100),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_model(model_path, device):
    model = MyCNN(input_channels=1, hidden_channels=[32, 64, 128], 
                use_batchnormalization=True, 
                num_classes=20, 
                kernel_size=[3, 3, 3])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert('L')  # Ensure the image is grayscale
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()



def main(config_file, image_paths):
    with open(config_file) as cf:
        config = json.load(cf)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(config["results_path"], "best_model.pth"), device)
    
    class_names = ['book', 'bottle', 'car', 'cat', 'chair', 'computermouse', 'cup', 'dog', 'flower', 'tree',
                   'fork', 'glass', 'glasses', 'headphones', 'knife', 'laptop', 'pen', 'plate', 'shoes', 'spoon']
    
    for image_path in image_paths:
        predicted_class = predict(model, image_path, transform, device)
        print(f"Image: {image_path} - Predicted Class: {class_names[predicted_class]}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    parser.add_argument("image_paths", type=str, nargs='+', help="Paths to images to be tested.")
    args = parser.parse_args()
    
    main(args.config_file, args.image_paths)
