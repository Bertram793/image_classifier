import torch
import torch.nn as nn
import timm


class SimpleFruitClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name="efficientnet_b0",
        pretrained=True
    ):
        super().__init__()

        backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.num_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)