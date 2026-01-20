import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioBaseline(nn.Module):
    def __init__(self, n_mfcc, n_classes, time_steps=94):
        super(AudioBaseline, self).__init__()
        # Deeper Conv1D with BatchNorm
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        # x: (B, n_mfcc, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2) # Extra pool
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x) # (B, 512, 1)
        x = x.squeeze(-1)       # (B, 512)
        return x


class FaceBaseline(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FaceBaseline, self).__init__()
        # Deeper 2D CNN with BatchNorm
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 4 pools: 128 -> 64 -> 32 -> 16 -> 8
        self.fc = nn.Linear(256 * 8 * 8, n_classes) 
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
        
    def extract_features(self, x):
        # x: (B, 1, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 64
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 32
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 16
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # 8
        
        x = x.view(x.size(0), -1) # (B, 256*8*8)
        return x
