"""
ASD Detection Model - ST-GCN based architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_GCN_Block(nn.Module):
    """Spatial-Temporal Graph Convolutional Block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.5):
        super().__init__()
        
        # Spatial graph convolution
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal convolution
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        # x: (batch, channels, frames, joints)
        res = self.residual(x)
        
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        
        x = x + res
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class ASD_Detection_Model(nn.Module):
    """
    Multi-task model for ASD detection and symptom classification
    
    Architecture:
    - Input: Skeleton sequences (batch, frames, joints, coords)
    - ST-GCN layers for feature extraction
    - Multi-task heads for:
      1. ASD detection (binary classification)
      2. Symptom classification (multi-class)
      3. Severity prediction (regression)
    """
    
    def __init__(self, num_symptoms=10, num_joints=24, in_channels=3):
        super().__init__()
        
        self.num_symptoms = num_symptoms
        self.num_joints = num_joints
        
        # ST-GCN layers
        self.st_gcn_layers = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, kernel_size=9),
            ST_GCN_Block(64, 64, kernel_size=9),
            ST_GCN_Block(64, 128, kernel_size=9),
            ST_GCN_Block(128, 128, kernel_size=9),
            ST_GCN_Block(128, 256, kernel_size=9),
            ST_GCN_Block(256, 256, kernel_size=9),
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension
        feature_dim = 256
        
        # ASD Detection Head (Binary Classification)
        self.asd_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Symptom Classification Head (Multi-class)
        self.symptom_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_symptoms)
        )
        
        # Severity Regression Head
        self.severity_regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Skeleton sequences (batch, frames, joints, coords)
        
        Returns:
            dict: {
                'asd_probability': ASD detection probability,
                'symptom_logits': Symptom classification logits,
                'severity': Severity score,
                'features': Extracted features
            }
        """
        batch_size = x.size(0)
        
        # Reshape to (batch, coords, frames, joints)
        x = x.permute(0, 3, 1, 2)
        
        # ST-GCN layers
        for layer in self.st_gcn_layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        features = x.view(batch_size, -1)  # (batch, 256)
        
        # Multi-task predictions
        asd_prob = self.asd_classifier(features)
        symptom_logits = self.symptom_classifier(features)
        severity = self.severity_regressor(features)
        
        return {
            'asd_probability': asd_prob,
            'symptom_logits': symptom_logits,
            'severity': severity,
            'features': features
        }

# Test the model
if __name__ == '__main__':
    print("🧪 Testing ASD Detection Model...")
    
    # Create model
    model = ASD_Detection_Model(num_symptoms=10, num_joints=24)
    
    # Create dummy input
    batch_size = 4
    frames = 150
    joints = 24
    coords = 3
    
    x = torch.randn(batch_size, frames, joints, coords)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    outputs = model(x)
    
    print(f"\nOutput shapes:")
    print(f"  ASD probability: {outputs['asd_probability'].shape}")
    print(f"  Symptom logits: {outputs['symptom_logits'].shape}")
    print(f"  Severity: {outputs['severity'].shape}")
    print(f"  Features: {outputs['features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✅ Model test complete!")