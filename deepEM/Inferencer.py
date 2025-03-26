from abc import ABC, abstractmethod
from typing import Any, List
import os
from pathlib import Path
import datetime
import torch

from deepEM.Utils import print_error, print_info, print_warning

class AbstractInference(ABC):
    """
    Abstract base class for model inference. Subclasses must implement all abstract methods
    to handle loading models, making predictions, and saving results.
    """
    def __init__(self, model_path: str, data_path: str) -> None:
        """
        Initialize the inference class with model and data paths.

        Args:
            model_path (str): Path to the model file.
            data_path (str): Path to the data directory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.find_model_file(model_path)
        self.data_path = data_path
        
        if(self.model_path):
            self.metadata = self.load_metadata()
            self.model = self.setup_model()
            self.load_checkpoint()

            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_to = os.path.join(self.data_path, f"results-{Path(self.model_path).parent.parent.parent.stem}", timestamp)
            os.makedirs(self.save_to, exist_ok=True)
            
            with open(os.path.join(self.save_to,"model-and-data.txt"), 'w') as file:
                file.write(f"Model path: {os.path.abspath(self.model_path)}\nData path: {os.path.abspath(self.data_path)}")
        
    def find_model_file(self, input_path):
        # Check if the input is a file
        if os.path.isfile(input_path):
            # Check if the file is "best_model.pth"
            if os.path.basename(input_path) == "best_model.pth":
                print_info(f"Found model checkpoint at {input_path}")
                return input_path
            elif(input_path.endswith('.pth')):
                print_warning("Cound not find a trained model: File is not named 'best_model.pth'. You should provide a file with the name 'best_model.pth' as this is the fully trained model saved at the best validation epoch.")
                return input_path
            else: 
                print_error(f"Could not find a trained model: The provided file {input_path} is not a model checkpoint (.pth).")
                return None
        # Check if the input is a directory
        elif os.path.isdir(input_path):
            # Recursively search for "best_model.pth" in the directory and subdirectories
            for root, dirs, files in os.walk(input_path):
                if "best_model.pth" in files:
                    print_info(f"Found model checkpoint at {input_path}")
                    return os.path.join(root, "best_model.pth")
            print_error("Cound not find a trained model: 'best_model.pth' not found in the directory or its subdirectories. Please provide the file directly or provide a directory which contains the file.")
            return None
        else:
            print_error(f"Cound not find a trained model: The provided path {input_path} is neither a valid file nor a directory.")
            return None
        
    def load_metadata(self) -> None:
        """
        Load the model weights and metadata from self.model_path
        metadata will be set as self.metadata

        Returns: 
            dict: metadata needed for inference
        """
        checkpoint = torch.load(self.model_path)
        metadata = checkpoint['metadata']
        return metadata
    
    def load_checkpoint(self) -> None:
        """
        Load the model weights and metadata from self.model_path
        metadata will be set as self.metadata

        Returns: 
            dict: metadata needed for inference
        """
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        self.metadata = checkpoint['metadata']
        return 
        
    
    def setup_model(self) -> None:
        """
        sets up the model class for inference.

        Returns: 
            torch.nn.Module: the model
        """
        raise NotImplementedError("The 'setup_model' method must be implemented by the DL specialist.")
    
    
    def get_image_files(self, folder_path):
        """
        Get all image files in the specified folder.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            list: List of image file paths.
        """
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif")  # Add more extensions if needed
        return [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(image_extensions)
        ]
        
        #[os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)]
    
    def load_images_from_folder(self, folder_path: str) -> List[Any]:
        """
        Load all images from a given folder.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            List[Any]: A list of loaded images.
        """
        paths = self.get_image_files(folder_path)
        print_info(f"Found {len(paths)} files to do inference.")
        images = []
        for path in paths:
            img = self.load_single_image(path)
            if(img is not None):
                images.append(img)
        return images
    
    def predict_batch(self, images: List[Any]) -> List[Any]:
        """
        Perform inference on a batch of images.

        Args:
            images (List[Any]): A list of input images.
        """
        # TODO add timings 
        
        for idx, image in enumerate(images):
            prediction = self.predict_single(image)
            self.save_prediction(image, prediction, os.path.join(self.save_to, f"prediction_{idx}"))
    
    def inference(self):
        # TODO add timings 
        with torch.no_grad():
            if(self.model_path):
                if(os.path.isdir(self.data_path)):
                    images = self.load_images_from_folder(self.data_path)
                    self.predict_batch(images)
                elif(os.path.isfile(self.data_path)):
                    image = self.load_single_image(self.data_path)
                    prediction = self.predict_single(images[0])
                    self.save_prediction(image, prediction, os.path.join(self.save_to, "prediction"))
                else: 
                    print_error(f"Path to data for inference does not exist. Could not find {self.data_path}. Is is neither a file nor a directory.")
                    return            
        return 
    



    @abstractmethod
    def predict_single(self, image: Any) -> Any:
        """
        Perform inference on a single image.

        Args:
            image (Any): The input image in raw format.

        Returns:
            Any: The prediction result for the image.
        """
        raise NotImplementedError("The 'predict_single' method must be implemented by the DL specialist.")

    

    @abstractmethod
    def save_prediction(self, input, prediction, save_file: str) -> None:
        """
        Save predictions to a file.

        Args:
            input (Any): single input to save.
            prediction (Any): Prediction of the input to save.
            save_file (str): Path to save the predictions.
        """
        raise NotImplementedError("The 'save_prediction' method must be implemented by the DL specialist.")


    @abstractmethod
    def load_single_image(self, img_file: str) -> List[Any]:
        """
        Load image from a given folder. This method should also implement possible preprocessing (like transforms if they are appled to to image before forwarding it to the model.)

        Args:
            img_file (str): Path to the image file.

        Returns:
            Any: the loaded image
        """
        raise NotImplementedError("The 'load_single_image' method must be implemented by the DL specialist.")
    
    
    

    