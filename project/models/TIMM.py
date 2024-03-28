import timm
import torch.nn as nn

class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, drop_rate: float = 0.1, pretrained: bool = True):
        super().__init__()
        self.name = model_name
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        data_config = timm.data.resolve_model_data_config(self.encoder)   
        self.transform = timm.data.create_transform(**data_config, is_training=pretrained, no_aug=True)
        self.num_classes = num_classes
        self.head = nn.Identity()
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(self.encoder.num_features, self.encoder.num_features), 
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(self.encoder.num_features, num_classes)
            )
        self.encoder_grad(True)
        self.head_grad(True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def classify(self, x):
        x = self.head(x)
        return x
    
    def encoder_grad(self, mode=True):
        for param in self.encoder.parameters():
            param.requires_grad = mode
    
    def head_grad(self, mode=True):
        for param in self.head.parameters():
            param.requires_grad = mode