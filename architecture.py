import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, input_channels: int, 
                 hidden_channels: list, 
                 use_batchnormalization: bool, 
                 num_classes: int, kernel_size: list, 
                 activation_function: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()
        
        layers = []
        
        # go through all the hidden channels
        for i in range(len(hidden_channels)):
            out_channels = hidden_channels[i]
            k_size = kernel_size[i]
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=k_size, padding=k_size//2))
            
            if use_batchnormalization:
                layers.append(nn.BatchNorm2d(out_channels))
                
            layers.append(activation_function) #activation function relu as default
            input_channels = out_channels #set output for input for next conv layer

        # prepare all layers and functions to be nicely used in forward
        self.hidden_layers = nn.Sequential(*layers)
        #self.output_layer = nn.Linear(hidden_channels[-1]*70*100, num_classes) #linear ouptut with fixed img size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Output size is (7, 7)
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2)  # Use max pooling with kernel size 2x2


        input_size = 100  # Change this to your actual input size
        pool_output_size = (input_size - 5) // 2 + 1 


        # self.output_layer = nn.Linear(hidden_channels[-1] * 7 * 7, num_classes)  # Adjusted to match pooled output size
        self.output_layer = nn.Linear(hidden_channels[-1] * pool_output_size * pool_output_size, num_classes)
        
        self.flatten = nn.Flatten()
        
        
        
        
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(input_images)
        # x = self.adaptive_pool(x)  # Ensure fixed output size
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        
        return x



