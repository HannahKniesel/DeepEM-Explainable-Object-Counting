import tifffile
from PIL import Image
from typing import Any, List
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM


from deepEM.Inferencer import AbstractInference

from src.Dataset import ResizeToMultipleWithLocations, patchify_image, stitch_patches
from src.Model import Model 


class SimpleDataset(Dataset):
    def __init__(self, image):
        self.patches, _, _, self.num_patches_x, self.num_patches_y = patchify_image(image)
        self.totensor = torchvision.transforms.ToTensor()
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.totensor(self.patches[idx])*255                   
        return patch, self.num_patches_x, self.num_patches_y
            
        
    
class Inference(AbstractInference):
    """
    Class for model inference. Implements all abstract methods
    to handle loading models, making predictions, and saving results.
    """
    def __init__(self, model_path: str, data_path: str) -> None:
        super().__init__(model_path, data_path)
        self.resize_transform = ResizeToMultipleWithLocations(224)

    def min_max_norm(self, value):
        return (value-value.min())/(value.max()-value.min())
    
    def load_single_image(self, img_file: str) -> List[Any]:
        """
        Load image from a given folder and preprocesses it. 

        Args:
            img_file (str): Path to the image file.

        Returns:
            Any: the loaded image
        """
        if img_file.lower().endswith((".tif", ".tiff")):
            with tifffile.TiffFile(img_file) as tif:
                properties = {}
                for tag in tif.pages[0].tags.values():
                    name, value = tag.name, tag.value
                    properties[name] = value
                image = tif.pages[0].asarray()
            return Image.fromarray(image) # Image.fromarray((self.min_max_norm(image)*255).astype(np.uint8))
        else:
            return None   
        
    def setup_model(self) -> None:
        """
        sets up the model class for inference.

        Returns: 
            torch.nn.Module: the model
        """
        return Model(self.metadata["class_names"])
     
    
    
    def predict_single(self, image: Any) -> Any:
        """
        Perform inference on a single image.

        Args:
            image (Any): The input image in raw format.

        Returns:
            Any: The prediction result for the image.
        """
        # make sure the image is a multiple of 224, such that we can patchify it.
        image,_ = self.resize_transform(image)
        dataset = SimpleDataset(image)
        dataloader = DataLoader(
            dataset,
            batch_size=16, 
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )
        
        # target_layers = [getattr(self.model.model.layer4, "2")]
        target_layers = [self.model.model.layer4[-1]] 
        
        img_patches = []
        cam_patches = []
        img_predictions = torch.zeros((1, len(self.metadata['class_names']))) # TODO adapt this to the models number of output neurons like self.model
        for batch in dataloader:
            patch, num_patches_x, num_patches_y = batch            
            with torch.enable_grad():
                with GradCAM(model=self.model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=patch, targets=None)
                outputs = cam.outputs
                predictions = outputs.detach().cpu().round()
                # Where the model predicts no virus, we set the gradCAM to 0
                grayscale_cam[predictions.sum(1).numpy() == 0] = 0
            img_patches.extend(patch.detach().cpu().numpy())
            cam_patches.extend(grayscale_cam.squeeze())
            img_predictions += torch.sum(predictions, dim = 0)
            
        full_image = stitch_patches(img_patches, num_patches_x=num_patches_x, num_patches_y=num_patches_y)
        full_cam = stitch_patches(cam_patches, num_patches_x=num_patches_x, num_patches_y=num_patches_y)
        
        return {"image": full_image, 
                "cam": full_cam, 
                "prediction": img_predictions.squeeze()}
            

    

    def save_prediction(self, input, prediction, save_file: str) -> None:
        """
        Save predictions to a file.

        Args:
            input (Any): single input to save.
            prediction (Any): Prediction of the input to save. (Return of the self.predict_single method)
            save_file (str): Filename and Path to save the predictions. You need to set the format.
        """
        fig, axs = plt.subplots(1, 2, figsize=(25, 10))
        axs[0].imshow(prediction["image"], cmap="gray")
        axs[0].set_title("Micrograph")
        axs[1].imshow(prediction["image"], cmap="gray")
        axs[1].imshow(prediction["cam"], alpha=0.5)
        axs[1].set_title("Micrograph with GradCAM")
        for a in axs:
            a.set_axis_off()

        title_str = f"Prediction: "
        for l, class_name in zip(prediction["prediction"], self.model.class_names): # TODO how to get the class names 
            title_str += f" {l.numpy().round()} {class_name} |"

        plt.suptitle(title_str)
        plt.tight_layout()
        plt.savefig(f"{save_file}.jpg")
        plt.close()
        print(f"Saved prediction to {save_file}")


    
    