import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AudioEfficientNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(AudioEfficientNet, self).__init__()
        # Load EfficientNet-B0
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        if pretrained:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.model = efficientnet_b0(weights=None)
            
        # EfficientNet input: 3 channels. We handle this in forward (expand).
        
        # Replace Classifier
        # EfficientNet classifier is .classifier (Sequential) -> [Dropout, Linear]
        # We replace the Linear layer.
        # Check in_features
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        
    def forward(self, x):
        # x: (B, 1, 40, 94) or (B, 40, 94)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

    def extract_features(self, x):
        import sys
        # Extract features (before classifier)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # print("DEBUG: Into Features", x.shape)
        x = self.model.features(x)
        # print("DEBUG: After Features", x.shape)
        x = self.model.avgpool(x)
        # print("DEBUG: After AvgPool", x.shape)
        x = torch.flatten(x, 1)
        # print("DEBUG: After Flatten", x.shape)
        # sys.stdout.flush()
        return x

class AudioResNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(AudioResNet, self).__init__()
        # Load ResNet18
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet18(weights=None)
            
        # Replace FC
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)
        
    def forward(self, x):
        # x shape: (B, 1, n_mfcc, T) -> (B, 1, 40, 94)
        # ResNet expects (B, 3, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # 1. Expand to 3 channels
        x = x.repeat(1, 3, 1, 1) # (B, 3, 40, 94)
        
        # 2. Resize? ResNet usually likes 224x224.
        # 40x94 is small. We can upsample.
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.model(x)
    
    def extract_features(self, x):
        # Need to hook or modify forward to return features before FC.
        # ResNet18 structure: .layer4 -> .avgpool -> .fc
        # Let's extract manually
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class FaceResNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(FaceResNet, self).__init__()
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet18(weights=None)
            
        # Input is grayscale (1 channel).
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        # x: (B, 1, 128, 128)
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

    def extract_features(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
