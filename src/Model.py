import os
import torch
from torchvision.models import resnet50
from copy import deepcopy
from pathlib import Path
from torchvision import transforms

from lib.Model import AbstractModel

weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained_models')

def create_model(class_names):
    num_classes = len(class_names)
    model = resnet50()

    # adapt the first conv layer to accept single channel grayscale image
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # adapt the fc layer for regression
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, num_classes, bias=True),
        torch.nn.Identity(),  # Linear activation, for the regression task
    )

    # load the pretrained CEM500k weights
    state = torch.load(
        Path(os.path.join(weights_dir, "cem500k_mocov2_resnet50_200ep.pth.tar")),
        map_location="cpu",
    )

    pretrained_norms = state["norms"]
    resnet50_state_dict = deepcopy(state["state_dict"])
    model.load_state_dict(resnet50_state_dict, strict=False)
    return model, pretrained_norms


class Model(AbstractModel):
    def __init__(self, class_names):
        """
        Initialize the AbstractModel class which extends torch.nn.Module.
        This class serves as a base class for model architectures.
        It defines methods for model forward pass, prediction, saving, and loading the model.
        """
        super(AbstractModel, self).__init__()
        self.model, pretrained_norms = create_model(class_names)
        self.class_names = class_names
        # Ensure mean and std are lists
        self.mean = [pretrained_norms[0]] if isinstance(pretrained_norms[0], float) else pretrained_norms[0]
        self.std = [pretrained_norms[1]] if isinstance(pretrained_norms[1], float) else pretrained_norms[1]
        
        self.norm_transform = transforms.Normalize(mean=self.mean, std=self.std, inplace=False)

        
        
    

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model's output.
        """
        x = self.norm_transform(x)
        return self.model(x)


        