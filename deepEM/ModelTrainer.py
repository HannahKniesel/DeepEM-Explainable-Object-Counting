import torch
import os
import copy
import math
import time
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, ConcatDataset

from deepEM.Utils import load_json, extract_defaults, get_fixed_parameters, format_time


config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
        
class AbstractModelTrainer(ABC):
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
        self.data_path = data_path
        self.logger = logger
        self.resume_from_checkpoint = resume_from_checkpoint
        self.parameter = get_fixed_parameters(os.path.join(config_dir,"parameters.json")) #load_json(os.path.join(config_dir,"parameters.json"))["parameter"]
        self.start_epoch = 0
        
        # det default hyperparameters
        hyperparameters = load_json(os.path.join(config_dir,"parameters.json"))
        self.parameter.update(extract_defaults(hyperparameters))
                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        trainset, valset, testset = self.setup_datasets()
        self.train_loader, self.val_loader, self.test_loader= self.setup_dataloaders(trainset, valset, testset)
        # setup number of training epochs 
        self.reduce_epochs = None
        self.set_epochs()
        self.finetuning = False
            
    def prepare(self, config=None, train_subset=None, reduce_epochs=None, set_parameters = True):
        if(set_parameters):
            if(not config):
                hyperparameters = load_json(os.path.join(config_dir,"parameters.json"))
                self.parameter.update(extract_defaults(hyperparameters))
                self.logger.log_warning(f"Could not find a config based on a hyperparameter search. Will use default parameters:\n{self.parameter}\n")
            else:
                self.parameter.update(config)
                
            self.logger.log_hyperparameters(config)
        self.train_subset = train_subset
        self.reduce_epochs = reduce_epochs
        
        if(not self.model):
            self.model = self.setup_model()
        
        # setup dataloaders
        trainset, valset, testset = self.setup_datasets()
        trainset = self.subsample_trainingdata(trainset)
        self.train_loader, self.val_loader, self.test_loader= self.setup_dataloaders(trainset, valset, testset)
        
        self.val_vis_loader, self.test_vis_loader = self.setup_visualization_dataloaders(valset, testset)
        
        # setup number of training epochs 
        self.set_epochs()
        
        # Set up optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer()
        
        # Track the best validation loss for early stopping and model checkpointing
        self.best_val_loss = math.inf
        self.patience_counter = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        
        
        
    
        
        
    def set_epochs(self):
        self.num_epochs = max([1,self.parameter["epochs"]])
        self.validation_interval = max([1,self.parameter['validation_interval']])
        if(self.reduce_epochs):
            self.num_epochs = max([1,int(self.reduce_epochs*self.num_epochs)])
            self.validation_interval = max([1,int(self.reduce_epochs*self.validation_interval)])
            
        
        
    def subsample_trainingdata(self, dataset): 
        if(self.train_subset is not None):
            num_samples = max([1,int(len(dataset)*self.train_subset)])
            torch.manual_seed(42)
            shuffled_indices = torch.randperm(len(dataset))[:num_samples]
            subset = torch.utils.data.Subset(dataset, shuffled_indices)
            return subset
        else: 
            return dataset
        
    @abstractmethod
    def setup_model(self):
        """
        Setup and return the model for training, validation, and testing.

        This method must be implemented by the DL expert.

        Returns:
            model (lib.Model.AbstractModel): The dataloader for the training dataset.
        """
        raise NotImplementedError("The 'setup_model' method must be implemented by the DL specialist.")
        
    @abstractmethod
    def setup_datasets(self):
        """
        Setup and return the datasets for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data_path provided by the EM specialist can b accessed via self.data_path

        Returns:
            trainset (torch.utils.data.Dataset): The dataloader for the training dataset.
            valset (torch.utils.data.Dataset): The dataloader for the validation dataset.
            testset (torch.utils.data.Dataset): The dataloader for the test dataset.
        """
        
        # Example implementation
        # # Define transforms (e.g., for normalization and augmentation)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        
        # train_dataset = CostumDataset(self.logger, self.data_dir, "train", data_format = ".tif", size_train_split = 0.6, annotations_dir = None, transform=transform)
        # val_dataset = CostumDataset(self.logger, self.data_dir, "val", data_format = ".tif", size_train_split = 0.6, annotations_dir = None, transform=transform)
        # test_dataset = CostumDataset(self.logger, self.data_dir, "test", data_format = ".tif", size_train_split = 0.6, annotations_dir = None, transform=transform)
        # return train_dataset, val_dataset, test_dataset
        
        # # Create dataloaders
        # train_loader = DataLoader(train_dataset, batch_size=self.parameter["batch_size"], shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=self.parameter["batch_size"], shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=self.parameter["batch_size"], shuffle=False)
        # return train_loader, val_loader, test_loader
        
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
        torch.manual_seed(42)
        shuffled_indices = torch.randperm(len(val_dataset))[:self.parameter["images_to_visualize"]]
        vis_val_subset = torch.utils.data.Subset(val_dataset, shuffled_indices)
        val_vis_loader = DataLoader(vis_val_subset, batch_size=int(self.parameter["batch_size"]), shuffle=False)
        
        shuffled_indices = torch.randperm(len(test_dataset))[:self.parameter["images_to_visualize"]]
        vis_test_subset = torch.utils.data.Subset(test_dataset, shuffled_indices)
        test_vis_loader = DataLoader(vis_test_subset, batch_size=int(self.parameter["batch_size"]), shuffle=False)
        return val_vis_loader, test_vis_loader
        
        
    def setup_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """
        Setup and return the dataloaders for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data_path provided by the EM specialist can b accessed via self.data_path
        
        Inputs:
            trainset (torch.utils.data.Dataset): The training dataset.
            valset (torch.utils.data.Dataset): The validation dataset.
            testset (torch.utils.data.Dataset): The test dataset.

        Returns:
            train_loader (torch.utils.data.DataLoader): The dataloader for the training dataset.
            valset_loader (torch.utils.data.DataLoader): The dataloader for the validation dataset.
            testset_loader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        """
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=int(self.parameter["batch_size"]), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(self.parameter["batch_size"]), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=int(self.parameter["batch_size"]), shuffle=False)
    

        return train_loader, val_loader, test_loader
          
        

    @abstractmethod
    def setup_optimizer(self):
        """
        Setup and return the optimizer and learning rate scheduler.

        This method must be implemented by the DL expert.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        """
        raise NotImplementedError("The 'setup_optimizer' method must be implemented by the DL specialist.")

    
    @abstractmethod
    def compute_loss(self, outputs, targets):
        """
        Compute the loss for a batch.
        
        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("The 'compute_loss' method must be implemented by the DL specialist.")
        

    @abstractmethod
    def train_step(self, batch):
        """
        Perform one training step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
        """
        # Implementation could look like this:
        # self.model.train()
        # inputs, targets = batch
        # self.optimizer.zero_grad()
        # outputs = self.model(inputs.to(self.device))
        # loss = self.compute_loss(outputs, targets.to(self.device))
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()
        raise NotImplementedError("The 'train_step' method must be implemented by the DL specialist.")
    
    def qualify(self, dataloader, file_name):
        idx = 0
        for batch in dataloader:
            images = self.visualize(batch)
            for img in images:
                idx +=1
                img = img.convert("RGBA")
                img.save(os.path.join(self.logger.samples_dir, f"{file_name}_{idx}.png"))
            
    def inference_metadata(self):
        """
        Returns possible metadata needed for inference (such as class names) as dictonary.
        This metadata will be saved along with model weights to the training checkpoints. 
        
        
        Returns:
            dict: dictonary with metadata
            
        """
        # Implementation could look like this:
        # metadata = {}
        # metadata["class_names"] = ["class1", "class2"]
        # return metadata
        # `return jit_model` 
        raise NotImplementedError("The 'inference_metadata' method must be implemented by the DL specialist.")
        
            
    def visualize(self, batch):
        """
        Visualizes the models input and output of a single batch and returns them as PIL.Image.

        Args:
            batch: A batch of data defined by your Dataset implementation.
        
        Returns:
            List[PIL.Image]: List of visualizations for the batch data.
            
        """
        # Implementation could look like this:
        # self.model.eval()
        # inputs, targets = batch
        # with torch.no_grad():
            # outputs = self.model(inputs)
            # loss = self.compute_loss(outputs, targets)
            # metrics = self.compute_metrics(outputs, targets)
        # return loss.item(), metrics
        raise NotImplementedError("The 'visualize' method must be implemented by the DL specialist.")
        

    def val_step(self, batch):
        """
        Perform one validation step.

        Args:
            batch: A batch of data defined by your Dataset implementation.
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        # Implementation could look like this:
        # self.model.eval()
        # inputs, targets = batch
        # with torch.no_grad():
            # outputs = self.model(inputs)
            # loss = self.compute_loss(outputs, targets)
            # metrics = self.compute_metrics(outputs, targets)
        # return loss.item(), metrics
        raise NotImplementedError("The 'val_step' method must be implemented by the DL specialist.")
        

    def test_step(self, batch):
        """
        Perform one test step.

        Args:
            batch: A batch of data defined by your Dataset implementation.
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        # Implementation could look like this:
        # self.model.eval()
        # inputs, targets = batch
        # with torch.no_grad():
        #     outputs = self.model(inputs)
        #     loss = self.compute_loss(outputs, targets)
        #     metrics = self.compute_metrics(outputs, targets)
        # return loss.item(), metrics
        raise NotImplementedError("The 'test_step' method must be implemented by the DL specialist.")
        

    def save_checkpoint(self, epoch, val_loss):
        """
        Save the best model checkpoint based on validation loss.
        
        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss.
        """
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss, 
            'parameter': self.parameter,
            'metadata':self.inference_metadata(),
        }
            
        # Save latest model for possible resuming of training
        checkpoint_path = os.path.join(self.logger.checkpoints_dir, f"latest_model.pth")
        torch.save(checkpoint, checkpoint_path)
        
        
        # Save best model for later use    
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0  # Reset the counter since we have improvement
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            checkpoint_path = os.path.join(self.logger.checkpoints_dir, f"best_model.pth")
        
            # Save model, optimizer, and scheduler state
            torch.save(checkpoint, checkpoint_path)

            
            # Save scripted model for easier exchange of model for inference. 
            checkpoint_path = os.path.join(self.logger.checkpoints_dir, "model_scripted.pt")
            self.model.eval()
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(checkpoint_path)
            
        else:
            self.patience_counter += 1

    def load_checkpoint(self, checkpoint_path, finetuning=False):
        """
        Load the model, optimizer, and scheduler state from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint to resume training from.
            finetuning (bool): Weather to use the checkpoint file for finetuing (will only set model weights, but ignore optimizer states and similar).

        """
        
        checkpoint = torch.load(checkpoint_path)
        self.parameter = checkpoint['parameter']
        self.prepare(set_parameters=False)
        
        
        self.model.load_state_dict(checkpoint['model_state_dict'])


        if(not finetuning):
            self.save_checkpoint(checkpoint['epoch'], checkpoint['val_loss'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.best_val_loss = checkpoint['val_loss']
            self.patience_counter = 0  # Reset patience counter
            self.start_epoch = checkpoint['epoch']
            self.logger.log_info(f"Resumed training from checkpoint: {checkpoint_path} (Validation Loss: {self.best_val_loss:.4f}) | Remaining epochs: {self.num_epochs - self.start_epoch}")
                        
        else: 
            self.start_epoch = 0
            self.logger.log_info(f"Loaded model checkpoint for finetuning from: {checkpoint_path} (Validation Loss: {self.best_val_loss:.4f})")
            
        

    def train_epoch(self, epoch):
        """
        Run one epoch of training and validation.
        
        Args:
            epoch (int): Current epoch.
        
        Returns:
            float: Average training loss for this epoch.
            float: Average validation loss for this epoch.
        """
        train_loss = 0.0
        val_loss = float("nan")
        metrics_sum = {}  # To store the sum of metrics across batches
        num_batches = len(self.val_loader)

        # Training loop
        self.model.train()
        for batch in self.train_loader:
            loss = self.train_step(batch)
            train_loss += loss
            
            # Step the learning rate scheduler if available
            if self.scheduler and (self.parameter['scheduler_step_by'] == "iteration"):
                self.scheduler.step()
        
        train_loss /= len(self.train_loader)

        # Validation loop
        
        with torch.no_grad():            
            if((epoch % self.validation_interval) == 0):
                val_loss = 0.0
                self.model.eval()
                self.qualify(self.val_vis_loader, f"validation_epoch-{epoch}")
                
                
                for batch in self.val_loader:
                    loss, batch_metrics = self.val_step(batch)
                    val_loss += loss

                    # Accumulate metrics from each batch
                    for metric, value in batch_metrics.items():
                        if metric not in metrics_sum:
                            metrics_sum[metric] = 0.0
                        metrics_sum[metric] += value

                # Calculate average validation loss
                val_loss /= num_batches

                # Calculate average for each metric
                metrics_avg = {metric: value / num_batches for metric, value in metrics_sum.items()}

                # Format metrics into a single log string
                metrics_str = ", ".join(f"{metric}: {avg_value:.4f}" for metric, avg_value in metrics_avg.items())

                # Save best model checkpoint and apply early stopping
                self.save_checkpoint(epoch, val_loss)

                # Early stopping logic
                if self.patience_counter >= self.parameter['early_stopping_patience']:
                    self.logger.log_info(f"Early stopping triggered at epoch {epoch}.")
                    return train_loss, val_loss, True  # Return True to indicate early stopping

        return train_loss, val_loss, False

    def test(self, evaluate_on_full):
        """
        Run the test loop after training.
        """
        self.logger.init(f"Evaluate")
        self.logger.init_directories()
        if(evaluate_on_full):
            # reinit dataset with no train/test_split
            combined_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset, self.test_loader.dataset])
            combined_dataloader = DataLoader(combined_dataset, batch_size=self.test_loader.batch_size, shuffle=False)
            self.test_loader = combined_dataloader
            self.logger.print_info(f"Evaluate on full dataset with {len(combined_dataset)} samples.")
        
        self.model.to(self.device)

        self.qualify(self.test_vis_loader, f"test")
        
        test_loss = 0.0
        metrics_sum = {}  # To store the sum of metrics across batches
        num_batches = len(self.test_loader)

        for batch in self.test_loader:
            loss, batch_metrics = self.test_step(batch)
            test_loss += loss

            # Accumulate metrics from each batch
            for metric, value in batch_metrics.items():
                if metric not in metrics_sum:
                    # metrics_sum[metric] = 0.0
                    metrics_sum[metric] = []
                    
                # metrics_sum[metric] += value
                metrics_sum[metric].append(value)
                

        # Calculate average test loss
        test_loss /= num_batches
        self.logger.log_info(f"Test loss: {test_loss:.4f}")

        # Calculate and log the average of each metric
        # metrics_avg = {metric: value / num_batches for metric, value in metrics_sum.items()}
        metrics_avg = {metric: np.mean(value) for metric, value in metrics_sum.items()}
        
        for metric, avg_value in metrics_avg.items():
            self.logger.log_info(f"{metric}: {avg_value:.4f}")

        self.logger.log_test_results(test_loss, metrics_avg)
        return test_loss

    def fit(self):
        """
        Train the model for a specified number of epochs.
        
        """
        self.logger.init_directories()
        # Resume training from checkpoint if specified, otherwise reset parameters before training
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint, self.finetuning)
        else: 
            self.model.reset_model_parameters_recursive()
            
        self.model.to(self.device)
        
        train_loss_history = []
        val_loss_history = [] 
        val_epoch = []
        train_epoch = []
        accum_time = 0
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), initial=self.start_epoch, total=self.num_epochs, desc=f"[Training Run] | Num Epochs: {self.num_epochs - self.start_epoch} | Dataset size: {len(self.train_loader.dataset)}"):
            start_time = time.time()
            train_loss, val_loss, early_stop = self.train_epoch(epoch)
            end_time = time.time()
            elapsed_time = end_time - start_time
            accum_time += elapsed_time
            remaining_time = (self.num_epochs - (epoch+1))*(accum_time/(epoch+1))
            # self.logger.log_info(f"Avg time single epoch: {format_time(accum_time/(epoch+1))} | Remaining time training: {format_time(remaining_time)}")
            
            train_loss_history.append(train_loss)
            train_epoch.append(epoch)
            
            if(not math.isnan(val_loss)):
                val_loss_history.append(val_loss)
                val_epoch.append(epoch)
                
            
            self.logger.plot_training_curves(train_loss_history, train_epoch, val_loss_history, val_epoch)

            if early_stop:
                break
            
            # Step the learning rate scheduler if available
            if self.scheduler and (self.parameter['scheduler_step_by'] == "epoch"):
                self.scheduler.step()
        self.logger.plot_training_curves(train_loss_history, train_epoch, val_loss_history, val_epoch, show = True)
        self.logger.log_info(f"Finished training. Find logs and model checkpoints at: {self.logger.log_dir}")
        return np.min(val_loss_history)


                