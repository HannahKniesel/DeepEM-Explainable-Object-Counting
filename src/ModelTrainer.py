
import os
import torch
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt
from PIL import Image
import io


from deepEM.ModelTrainer import AbstractModelTrainer
from src.Model import Model 
from src.Dataset import TrainDataset, InferenceDataset

mse_loss = torch.nn.MSELoss()

class ModelTrainer(AbstractModelTrainer):
    def __init__(self, data_path, logger, resume_from_checkpoint = None):
        """
        Initializes the trainer class for training, validating, and testing models.

        Args:
            model (torch.nn.Module): The model to train.
            logger (Logger): Logger instance for logging events.
            config (dict): Contains all nessecary hyperparameters for training. Must at least contain: `epochs`, `early_stopping_patience`, `validation_interval`, `scheduler_step_by`.
            resume_from_checkpoint (str): Path to resume checkpoint
            train_subset (float, optional): Use subset of training data. This can be used for quick hyperparamtertuning. Defaults to `None`. 
            reduce_epochs (float, optional): Use subset of epochs. This can be used for quick hyperparamtertuning. Defaults to `None`. 
        """
        super().__init__(data_path, logger, resume_from_checkpoint )
        
    def setup_model(self):
        """
        Setup and return the model for training, validation, and testing.

        This method must be implemented by the DL expert.

        Returns:
            model (lib.Model.AbstractModel): The dataloader for the training dataset.
        """
        annotation_file = os.path.join(self.data_path,"annotations.xml")
        root = ET.parse(annotation_file).getroot()
        label_names = [label.find('name').text.lower() for label in root.findall('.//task/labels/label')]
        return Model(label_names)

    
    def inference_metadata(self):
        """
        Returns possible metadata needed for inference (such as class names) as dictonary.
        This metadata will be saved along with model weights to the training checkpoints. 
        
        
        Returns:
            dict: dictonary with metadata
            
        """
        metadata = {}
        metadata["class_names"] = self.val_loader.dataset.class_names
        return metadata
        
            
    def setup_datasets(self):
        """
        Setup and return the dataloaders for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data_path provided by the EM specialist can b accessed via self.data_path

        Returns:
            train_loader (torch.utils.data.DataLoader): The dataloader for the training dataset.
            val_loader (torch.utils.data.DataLoader): The dataloader for the validation dataset.
            test_loader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        """
        
        train_dataset = TrainDataset(os.path.join(self.data_path,"annotations.xml"), self.data_path, "train")
        val_dataset = InferenceDataset(os.path.join(self.data_path,"annotations.xml"), self.data_path, "val")
        test_dataset = InferenceDataset(os.path.join(self.data_path,"annotations.xml"), self.data_path, "test")
       
        return train_dataset, val_dataset, test_dataset
    
    def setup_visualization_dataloaders(self, val_dataset, test_dataset):
        """
        Setup and return the dataloaders for visualization during validation, and testing.
        This method will subsample the val_dataset and test_dataset to contain self.parameter["images_to_visualize"] datapoints
        This method should be overidden for imbalanced data, to pick the most interesting data samples.
                        
        Inputs:
            valset (torch.utils.data.Dataset): The validation dataset.
            testset (torch.utils.data.Dataset): The test dataset.

        Returns:
            val_vis_loader (torch.utils.data.DataLoader): The dataloader for visualizing a subset of the validation dataset.
            test_vis_loader (torch.utils.data.DataLoader): The dataloader for visualizing a subset of the test dataset.
        """
        indices_val = []
        for idx,item in enumerate(val_dataset):
            _, _, targets, _, _, _, _ = item
            if(torch.sum(targets)>0):
                indices_val.append(idx)
            if(len(indices_val) == self.parameter["images_to_visualize"]):
                break
        vis_val_subset = torch.utils.data.Subset(val_dataset, indices_val)
        val_vis_loader = DataLoader(vis_val_subset, batch_size=self.parameter["batch_size"], shuffle=False)
        
        indices_test = []
        for idx,item in enumerate(test_dataset):
            _, _, targets, _, _, _, _ = item
            if(torch.sum(targets)>0):
                indices_test.append(idx)
            if(len(indices_test) == self.parameter["images_to_visualize"]):
                break
        vis_test_subset = torch.utils.data.Subset(test_dataset, indices_test)
        test_vis_loader = DataLoader(vis_test_subset, batch_size=self.parameter["batch_size"], shuffle=False)
        return val_vis_loader, test_vis_loader
          
        

    def setup_optimizer(self):
        """
        Setup and return the optimizer and learning rate scheduler.

        This method must be implemented by the DL expert.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.parameter["learning_rate"])
        lr_scheduler = None
        return optimizer, lr_scheduler

    
    def compute_loss(self, outputs, targets):
        """
        Compute the loss for a batch.
        
        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        return mse_loss(outputs, targets)
        

    def train_step(self, batch):
        """
        Perform one training step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
        """
        img, value = batch
        self.optimizer.zero_grad()
        outputs = self.model(img.to(self.device))
        loss = self.compute_loss(outputs, value.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_gradcam(self, img):
        target_layers = [self.model.model.layer4[-1]] 
        with torch.enable_grad():
            with GradCAM(model=self.model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=img[None,...], targets=None)[0]
        return grayscale_cam
    
    def visualize(self, batch):
        """
        Visualizes the models input and output of a single batch and returns them as PIL.Image.

        Args:
            batch: A batch of data defined by your Dataset implementation.
        
        Returns:
            List[PIL.Image]: List of visualizations for the batch data.
            
        """
        self.model.eval()
        imgs, normed_imgs, targets, _, _, _, idx = batch
        outputs = self.model(normed_imgs.to(self.device)).detach().cpu()
        pil_images = []
        for img, normed_img, target, out in zip(imgs, normed_imgs, targets, outputs):
            cam = self.get_gradcam(normed_img)
            # TODO plot annotated locations?            


            fig, axs = plt.subplots(1, 2, figsize=((2*5, 5)))
            axs[0].imshow(img.squeeze(), cmap="gray")
            axs[0].set_axis_off()
            axs[1].imshow(img.squeeze(), cmap="gray")
            axs[1].imshow(cam.squeeze(), alpha=0.5)
            axs[1].set_title(f"GradCAM overlay")    
            axs[1].set_axis_off()

            title_str = f"Prediction: "
            for l, class_name in zip(out, self.val_loader.dataset.class_names): 
                title_str += f" {l.numpy().round()} {class_name} |"
            title_str += "\n Label"
            for o, class_name in zip(target, self.val_loader.dataset.class_names): 
                title_str += f" {o.numpy().round()} {class_name} |"

            plt.suptitle(title_str)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='PNG')
            buf.seek(0)  # Move the cursor to the beginning of the buffer
            pil_image = Image.open(buf)
            pil_images.append(pil_image)
            plt.close()
        return pil_images
    
    def compute_metrics(self, outputs, targets):
        # outputs have shape (bs,num_classes), targets have shape  (bs, num_classes)
        outputs = outputs.round().int().cpu()        
        mae = torch.abs(outputs - targets).mean().item()
        
        metrics = {"MAE": mae}
        
        num_classes = outputs.shape[-1]
        class_names = self.val_loader.dataset.class_names
        for n in range(num_classes):
            class_o = outputs[:,n].cpu()
            class_t = targets[:,n].cpu()
            metrics[f"MAE-class-{class_names[n]}"] = torch.abs(class_o - class_t).mean().item()
            # TODO Add more metrics      
            
        return metrics
        

    def val_step(self, batch):
        """
        Perform one validation step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        self.model.eval()
        img, normed_img, targets, num_patch_x, num_patch_y, img_idx, idx = batch
        outputs = self.model(normed_img.to(self.device))
        loss = self.compute_loss(outputs, targets.to(self.device))
        metrics = self.compute_metrics(outputs, targets)
        return loss.item(), metrics
        

    def test_step(self, batch):
        """
        Perform one test step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        # Implementation could look like this:
        return self.val_step(batch)

    