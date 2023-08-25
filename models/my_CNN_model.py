"""
This file declare a sample CNN models follow with FNN layers
"""
import torch
import torch.nn as nn
from torchsummary import summary

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, final_ReLu=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.ReLU = nn.ReLU(inplace=True)
        self.final_ReLu = final_ReLu

    def forward(self, x):
        x = self.double_conv(x)
        if self.final_ReLu: 
            x = self.ReLU(x)
        return x
    

    
class My_CNN_models(nn.Module):
    def __init__(self, output_dims):
        super(My_CNN_models, self).__init__()

        self.layer_1 = nn.Sequential(DoubleConv(3,64), 
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_2 = DoubleConv(64,64, final_ReLu=False)
        self.layer_3 = nn.Sequential(DoubleConv(64,128), 
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_4 = DoubleConv(128,128, final_ReLu=False)
        self.layer_5 = nn.Sequential(DoubleConv(128,256), 
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_6 = DoubleConv(256,256, final_ReLu=False)
        self.layer_7 = nn.Sequential(DoubleConv(256,512), 
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer_8 = DoubleConv(512,512, final_ReLu=False)
        self.layer_last = nn.Sequential(DoubleConv(512,512),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        DoubleConv(512,512),
                                        nn.MaxPool2d(kernel_size=3, stride=2))

        self.ReLu = nn.ReLU(inplace=True)

        self.outLayer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, output_dims),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x) + x
        x = self.ReLu(x)
        x = self.layer_3(x)
        x = self.layer_4(x) + x
        x = self.ReLu(x)
        x = self.layer_5(x)
        x = self.layer_6(x) + x
        x = self.ReLu(x)
        x = self.layer_7(x)
        x = self.layer_8(x) + x
        x = self.ReLu(x)
        x = self.layer_last(x)

        x = x.view(x.size(0), -1)
        x = self.outLayer(x)
        return x

if __name__ == '__main__':
    # Create an instance of the model
    model = My_CNN_models(50)

    # dummy input
    dummy_input = torch.rand((1,3,224,224))

    output = model(dummy_input)

    print(output)

    # Print the model summary
    summary(model, input_size=(3, 224, 224))


    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print(f"count second method: {pytorch_total_params}")