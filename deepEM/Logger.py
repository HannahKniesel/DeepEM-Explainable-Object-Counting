import os
import datetime
import json
import matplotlib.pyplot as plt
import torch
import psutil
import logging
import sys
from pathlib import Path
import re
import numpy as np

from deepEM.Utils import load_json


class Logger:
    def __init__(self, data_path):
        """
        Initialize the logger, creating a directory to store logs.
        """
        self.data_path = data_path
        # Create a timestamped directory for logs
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{Path(data_path).stem}_{timestamp}"
        self.root_dir = os.path.join("logs", name)
        self.init()
        
    def init(self, file_name=None):
        """
        Initialize the logger, creating a directory to store logs.
        """
        # Create a timestamped directory for logs
        if(file_name):
            self.log_dir = os.path.join(self.root_dir, file_name)
        else:
            self.log_dir = self.root_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Subdirectories for specific logs
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.plots_dir = os.path.join(self.log_dir, "plots")
        self.samples_dir = os.path.join(self.log_dir, "samples")


        # Set up the logger for info, warning, and error logging
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to prevent duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
    
        # Create file handler for logging to file
        log_file_path = os.path.join(self.log_dir, "log.txt")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Create stream handler for logging to stdout and stderr
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        # Define formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.log_info(f"Logger initialized. Logs will be saved to: {self.log_dir}")

        
    def init_directories(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        return True
        
    def log_info(self, message):
        """
        Log an info message both to stdout and to a log file.
        
        Args:
            message (str): The message to log.
        """
        self.logger.info(message)
        #print(f"[INFO] {message}")

    def log_warning(self, message):
        """
        Log a warning message both to stdout and to a log file.
        
        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)
        #print(f"[WARNING] {message}")

    def log_error(self, message):
        """
        Log an error message both to stderr and to a log file.
        
        Args:
            message (str): The message to log.
        """
        self.logger.error(message)
        
            
    def log_sweepparameters(self, hyperparams, val_loss):
        """
        Save hyperparameters to a JSON file.
        
        Args:
            hyperparams (dict): Dictionary of hyperparameters.
        """
        hyperparams_path = os.path.join(self.data_path, "Sweep_Parameters", "best_sweep_parameters.json")
        updated = False
        try:
            sweep_log = load_json(hyperparams_path)
            sweep_log["tested_configurations"].append(hyperparams.copy())
            if(val_loss < sweep_log["best_params"]["val_loss"]):
                hyperparams['val_loss'] = val_loss
                sweep_log["best_params"] = hyperparams
                updated = True
        except: 
            sweep_log = {}
            sweep_log["tested_configurations"] = [hyperparams.copy()]
            hyperparams['val_loss'] = val_loss
            sweep_log["best_params"] = hyperparams
            os.makedirs(os.path.join(self.data_path, "Sweep_Parameters"))
            
        with open(hyperparams_path, "w") as f:
            json.dump(sweep_log, f, indent=4)
        self.log_info(f"Current best sweep parameters were saved to {hyperparams_path}")
        return updated
        
            
    def load_best_sweep(self):
        hyperparams_path = os.path.join(self.data_path, "Sweep_Parameters", "best_sweep_parameters.json")
        try: 
            return load_json(hyperparams_path)["best_params"]
        except:
            return None
        
    def check_if_sweep_exists(self, hyperparams):
        hyperparams_path = os.path.join(self.data_path, "Sweep_Parameters", "best_sweep_parameters.json")
        try: 
            tested_configurations = load_json(hyperparams_path)["tested_configurations"]
            # exists = (hyperparams in tested_configurations)
            exists = any(hyperparams == config for config in tested_configurations)
            if(exists): 
                self.log_info(f"Current sweep configuration {hyperparams} already exists at {hyperparams_path}. Continue to next configuration.")
            return exists
        except:
            self.log_info(f"Could not find a sweep configuration at {hyperparams_path}.")
            return False
        
    

    def log_hyperparameters(self, hyperparams):
        """
        Save hyperparameters to a JSON file.
        
        Args:
            hyperparams (dict): Dictionary of hyperparameters.
        """
        hyperparams_path = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
        self.log_info(f"Hyperparameters saved to {hyperparams_path}")

    def save_checkpoint(self, model, val_loss, epoch):
        """
        Save the best-performing model checkpoint based on validation loss.
        
        Args:
            model (torch.nn.Module): The model to save.
            val_loss (float): Validation loss.
            epoch (int): Current epoch.
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, f"best_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        self.log_info(f"Checkpoint saved: {checkpoint_path} (Validation Loss: {val_loss:.4f})")

    def plot_training_curves(self, train_loss, train_epoch, val_loss, val_epoch, show = False):
        """
        Plot and save training and validation loss curves.
        
        Args:
            train_loss (list): Training loss history.
            val_loss (list): Validation loss history.
        """
        plt.figure()
        plt.plot(train_epoch, train_loss, label="Train Loss")
        plt.plot(val_epoch, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        
        plot_path = os.path.join(self.plots_dir, "training_curves.png")
        if(show):
            plt.show()
        else:
            plt.savefig(plot_path)
            plt.close()

    def log_test_metrics(self, metrics):
        """
        Log test metrics to a JSON file.
        
        Args:
            metrics (dict): Dictionary of test metrics.
        """
        metrics_path = os.path.join(self.log_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.log_info(f"Test metrics saved to {metrics_path}")


    def get_resource_usage(self):
        """
        Log system and GPU resource usage.
        
        """
        # Log CPU and memory usage
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        ram_usage = memory_info.used / (1024 ** 3)  # Convert bytes to GB
        total_ram = memory_info.total / (1024 ** 3)

        # Log GPU usage (if available)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            gpu_memory_allocated = gpu_memory_total = 0
            
        resources = {"CPU": cpu_percent, 
                     "RAM": ram_usage, 
                     "Total RAM": total_ram, 
                     "GPU": gpu_memory_allocated, 
                     "Total GPU": gpu_memory_total}
        
        resources_str = f"CPU: {cpu_percent}%, RAM: {ram_usage:.2f}GB/{total_ram:.2f}GB, GPU: {gpu_memory_allocated:.2f}GB/{gpu_memory_total:.2f}GB\n"
        return resources, resources_str
    
    def log_resource_usage(self):
        # Save resource usage to a log file
        resources, resources_str = self.get_resource_usage()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        resource_log_path = os.path.join(self.log_dir, "resource_usage.log")
        with open(resource_log_path, "a") as f:
            f.write(
                f"{timestamp}: {resources_str}"
            )
            
    def log_test_results(self, test_loss, metrics):
        save_to = os.path.join(self.log_dir, "test_results.txt")
        result_str = f"Test loss: {test_loss:.4f}\n"
        for metric in metrics.keys(): 
            result_str += f"{metric}: {metrics[metric]:.4f}\n"   
            
        with open(save_to, "w") as f:
            f.write(result_str)
        return
               
            
            
    def save_sample_images(self, **kwargs):
        """
        Save qualitative visualization of sampled images.

        This method must be implemented by the DL specialist.

        Args:
            **kwargs: Keyword arguments containing necessary parameters for image visualization.
        """
        raise NotImplementedError("This method must be implemented by the DL specialist.")




    def get_most_recent_logs(self):
        """
        Retrieves the most recent log directories for each dataname in the folder.
    
        Returns:
            dict: A dictionary where keys are datanames and values are paths to the most recent log directories.
        """
        # Dictionary to store the most recent log directory for each dataname
        most_recent_logs = {}
    
        # Path to the logs folder
        folder_path = "./logs/"
    
        # Regular expression to match the directory structure
        pattern = re.compile(r"(?P<dataname>.+)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
    
        for dir_name in os.listdir(folder_path):
            match = pattern.match(dir_name)
            if match:
                dataname = match.group("dataname")
                timestamp_str = match.group("timestamp")
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
    
                # Update the most recent log for the dataname
                if dataname not in most_recent_logs or timestamp > most_recent_logs[dataname]["timestamp"]:
                    most_recent_logs[dataname] = {
                        "timestamp": timestamp,
                        "path": os.path.join(folder_path, dir_name),
                    }
    
        # Extract only the paths from the dictionary
        return {dataname: info["path"] for dataname, info in most_recent_logs.items()}





    # def save_sample_images(self, **kwargs):
    #     """
    #     Custom implementation for saving qualitative visualization of sampled images.

    #     Args:
    #         **kwargs: Keyword arguments containing necessary parameters for image visualization.
    #             - 'images': List of image tensors to visualize.
    #             - 'title': Title for the visualization.
    #             - 'filename': Filename to save the visualization.
    #             - Any other arguments needed for customization.
    #     """
    #     images = kwargs.get('images', [])
    #     title = kwargs.get('title', 'Sample Images')
    #     filename = kwargs.get('filename', 'sample_images.png')
        
    #     if not images:
    #         raise ValueError("No images provided for visualization.")
        
    #     # Custom implementation for saving images (for example, with grid or other formatting)
    #     fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    #     for i, img in enumerate(images):
    #         axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
    #         axes[i].axis("off")
    #     fig.suptitle(title)
        
    #     sample_path = os.path.join(self.samples_dir, filename)
    #     plt.savefig(sample_path)
    #     plt.close()
    #     self.log_info(f"Sample images saved to {sample_path}")