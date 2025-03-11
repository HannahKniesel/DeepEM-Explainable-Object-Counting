import torch

# we recommend to do inference transforms (like normalization) within the models forward pass. This way, these transforms will be saved with the model itself and inference can be easily done.
class AbstractModel(torch.nn.Module):
    def __init__(self):
        """
        Initialize the AbstractModel class which extends torch.nn.Module.
        This class serves as a base class for model architectures.
        It defines methods for model forward pass, prediction, saving, and loading the model.
        """
        super(AbstractModel, self).__init__()
        
    def reset_model_parameters_recursive(self):
        """
        Recursively reset the parameters of all layers in a PyTorch model.
        """
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model's output.
        """
        raise NotImplementedError("The 'forward' method must be implemented by the DL specialist.")

    def predict(self, x):
        """
        Make predictions with the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted output.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            return self.forward(x)
        
        

        