import abc
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from glob import glob



class EMDataset(Dataset, abc.ABC):
    def __init__(self, logger, data_dir, split, data_format = ".tif", size_train_split = 0.6, annotations_dir = None, transform=None):
        """
        Args:
            logger (callable): Logger class.
            data_paths (list): List of file paths to the data.
            split (str): Data split. Should be one of `train`, `val`, `test`. 
            size_train_split (float, optional): The size of the training split. Defaults to 0.6 (60%) of the data.
            annotation_dir (str, optional): Directory to annotations.
            transform (callable, optional): Transform to apply to samples.
        """
        self.logger = logger
        data_paths = glob(data_dir+"*"+data_format)
        
        if(not self.check_datastructure(self, data_paths, annotations_dir)):
            logger.log_error("Datastructure is not as expected. Please check again, if the provided data is structured as defined in `1.4. Data Structuring`.")
        
        self.logger.log_info("Preprocess data...")
        self.data_paths = self.preprocess(data_paths)
        
        # split data into train/val/test set
        train, rest = train_test_split(self.data_paths, test_size=(1-size_train_split), random_state=42)
        val, test = train_test_split(rest, test_size=0.5, random_state=42)
        if split == "train":
            self.data_paths = train
        elif split == "val":
            self.data_paths = val
        elif split == "test":
            self.data_paths = test
        else:
            print(f"{split} not implemented. Please try 'train', 'val' or 'test'.")

        self.annotations_dir = annotations_dir
        self.transform = transform

    def __len__(self):
        """Return the total number of data samples."""
        return len(self.data_paths)

    def __getitem__(self, index):
        """
        Retrieve the `index`-th sample from the dataset.
        
        By default:
        - It loads the raw data from the file path specified by `self.data_paths[index]`.
        - Applies any preprocessing defined in `load_data`.
        - Optionally applies a transformation if `self.transform` is specified.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            object: The processed sample (e.g., image or tensor). The exact return type
            depends on the implementation of `load_data` and `self.transform`.

        Notes:
            - This method can be overridden to customize sample retrieval.
            - `load_data` must be implemented by the subclass.
        """
        # Retrieve the file path at the given index
        file_path = self.data_paths[index]
        
        # Load raw data (subclass must implement this method)
        img, value = self.load_data(file_path)
        
        # Apply transformation, if any
        if self.transform:
            img = self.transform(img)
        
        return img, value


    def preprocess(self, data_paths):
        """
        Preprocess the input data (e.g., denoising, contrast enhancing, etc.).
        If implemented, this method should save the preprocessed data to disk 
        and return the file paths to the preprocessed data.
        This method should check if the data was already preprocessed by 
        checking if the data was already written to disk. If so 
        it can be skipped.
        
        Args:
            data_paths (list): List of file paths to the raw input data.

        Returns:
            list: List of file paths to the preprocessed data. By default, this is the input `data_paths`.
        """
        # Default behavior: no preprocessing, return input file paths unchanged
        return data_paths

    def postprocess(self, predictions, **kwargs):
        """
        Postprocess model predictions to reconstruct the original image or volume.

        This is a flexible method where DL specialists can define all necessary parameters 
        (except `predictions`) themselves. The base implementation serves as an example 
        for typical postprocessing (e.g., patch stitching).

        Args:
            predictions (list): A list of predictions for individual patches. Each prediction
                should be a tensor or array.
            **kwargs: Additional parameters for custom postprocessing logic. These might include:
                - original_shape (tuple): The shape (height, width) of the original image.
                - patch_positions (list): A list of (x, y) positions for patch placement.
                - patch_size (tuple): The size of each patch. Defaults to input `predictions`.

        Returns:
            torch.Tensor or np.ndarray: The reconstructed full-sized image or volume.
        """
        return predictions


    @abc.abstractmethod
    def load_data(self, file_path):
        """
        Load raw data and optionally its corresponding annotation from the specified file path.
        Must be implemented by the subclass.
        
        Args:
            file_path (list): file paths to the input data.

        Returns:
            img: The loaded image-data to be forwarded to the self.transforms. Should be PIL.Image or torch.tensor.
            value: The label (as value) of the the loaded image used for loss computations. Should be torch.tensor.

        """
        raise NotImplementedError("The `load_data` method must be implemented by the subclass.")


    @abc.abstractmethod
    def check_datastructure(self):
        """
        Check that the provided data by EM specialists follows your expected and documented datastructure.
        
        Args:
            file_path (list): file paths to the input data.

        Returns:
            is_correct: boolean if the provided data follows the defined structure.
        
        """
        raise NotImplementedError("The `check_datastructure` method must be implemented by the subclass.")
