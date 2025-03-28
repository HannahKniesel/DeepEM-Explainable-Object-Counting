import json
import os
import time
import random
from math import prod  # For Python 3.8+
from itertools import product
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

from deepEM.Utils import format_time, load_json, extract_defaults

config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')


class ModelTuner(ABC):
    def __init__(self, model_trainer, data_path, logger):
        self.model_trainer = model_trainer
        self.data_path = data_path
        self.logger = logger
        self.config = self.load_config(os.path.join(config_dir,"parameters.json"))
        self.trainsubset = float(self.config["train_subset"])
        self.reduce_epochs = float(self.config["reduce_epochs"])
        self.method = self.config["method"]
        if(self.method not in ["grid", "random", "bayes"]):
            self.logger.log_warning(f"Method {self.method} is not in default methods. Please provide one of 'grid', 'random', 'bayes'.")
        # TODO resume sweep
        # TODO implement timing

        
    def edit_hyperparameters(self):
        hyperparameters = self.logger.load_best_sweep()
        if(hyperparameters):
            title = widgets.HTML(f"<p>Found best hyperparameters (val_loss = {hyperparameters['val_loss']:.4f}) for current dataset ({Path(self.logger.data_path).stem}).</p>")
        else: 
            title = widgets.HTML(f"<p>Could not find best hyperparameters for current dataset ({Path(self.logger.data_path).stem}). Make sure to conduct a hyperparameter sweep for each dataset for the best possible performance. To do so, you can execute the cells above.<p>")
            hyperparameters = load_json(os.path.join(config_dir,"parameters.json"))
            hyperparameters = extract_defaults(hyperparameters)
        
        widgets_list = [title]
        for k in hyperparameters.keys():
            if(k != "val_loss"):
                widgets_list.append(widgets.Text(value=str(hyperparameters[k]), description=f'{k}'))
        return widgets.VBox(widgets_list)
     
    def update_hyperparameters(self, widget_box):
        """Update the JSON configuration based on the values in the widget form."""
        children = widget_box.children
        index = 1  # Starting index for hyperparameters (after general parameters and <hr>)
        parameters = {}
        for child in children[index:]:
            v = child.value
            parameters[child.description] = float(v) if ('.' in v) or ('e' in v) else int(v) 
        return parameters    
    
    
    def update_config(self, widget_box):
        """Update the JSON configuration based on the values in the widget form."""
        children = widget_box.children
        index = 7  # Starting index for hyperparameters (after general parameters and <hr>)
        
        self.trainsubset = children[2].children[0].value/100 
        self.reduce_epochs = children[3].children[0].value/100

        for param, details in self.config['hyperparameter'].items():
            values_widget = children[index + 2]  # Values widget for each parameter
            values = values_widget.value

            # Convert string of comma-separated values back to list of appropriate types
            try:
                parsed_values = [float(v) if ('.' in v) or ('e' in v) else int(v) for v in values.split(',')]
            except ValueError:
                raise ValueError(f"Invalid values provided for parameter '{param}'. Ensure all values are numeric.")

            self.config['hyperparameter'][param]['values'] = parsed_values
            index += 4  # Each parameter has 4 widgets (explanation, default, values, <br>)

        return self.config

    def load_config(self, config_file):
        """Load the hyperparameter configuration from a JSON file."""
        with open(config_file, "r") as f:
            return json.load(f)
        
        

    def get_default_params(self):
        """Extract default hyperparameters from the config."""
        return {key: value["default"] for key, value in self.config["hyperparameter"].items()}

    def prepare_grid_search_space(self):
        """Prepare the search space for grid search."""
        return {key: value["values"] for key, value in self.config["hyperparameter"].items()}


    def tune_grid(self):
        """Perform grid search tuning."""
        search_space = self.prepare_grid_search_space()
        best_sweep = self.logger.load_best_sweep()
        best_index = -1
        if(best_sweep is not None):
            best_params = {k: v for k, v in best_sweep.items() if k != "val_loss"}
            best_loss = best_sweep["val_loss"]
            self.logger.log_info(f"Found sweep log with current best parameters: {best_params}")
        else: 
            best_params, best_loss = None, float("inf")

        total_combinations = prod(len(v) for v in search_space.values())
        
        accum_time = 0
        for index,params in enumerate(product(*search_space.values())):
            self.logger.init(f"Sweep_{index}")
            self.logger.log_info(f"Start Sweep {index+1} of {total_combinations}...")
            hyperparams = dict(zip(search_space.keys(), params))
            self.logger.log_info(f"Current hyperparams {hyperparams}")
            if(self.logger.check_if_sweep_exists(hyperparams)): 
                continue
            
            try:
                print(f"Train subset: {self.trainsubset}")
                self.model_trainer.prepare(hyperparams, self.trainsubset, self.reduce_epochs) 
                start_time = time.time()
                val_loss = self.model_trainer.fit()
                end_time = time.time()
                elapsed_time = end_time - start_time
                accum_time += elapsed_time
                remaining_time = (total_combinations - (index+1))*(accum_time/(index+1))
                self.logger.log_info(f"Hyperparameters: {hyperparams}, Validation Loss: {val_loss}")
                self.logger.log_info(f"Avg time single sweep: {format_time(accum_time/(index+1))} | Remaining_time: {format_time(remaining_time)}")
            except Exception as e:
                self.logger.log_error(f"An error occurred during hyperparameter search with following parameters: \n{hyperparams}")
                self.logger.log_error(f"Error: \n{e}\n")
                
            updated = self.logger.log_sweepparameters(hyperparams, val_loss)
            if(updated):
                best_index = index                

        if(best_index == -1):
            self.logger.log_info(f"Best Parameters: {best_params}, Best Loss: {best_loss}, from previous sweep.")
        else:
            self.logger.log_info(f"Best Parameters: {best_params}, Best Loss: {best_loss}, Best Sweep index: {best_index}")
        return best_params, best_loss
    

    def tune(self):
        self.logger.log_info("Start hyperparameter sweep...")
        
        """method for tuning to be implemented by DL specialists."""
        if(self.method == "grid"):
            best_params, best_loss = self.tune_grid()
            self.logger.log_info(f"Finished sweep with best validation loss = {best_loss}.")
            self.logger.log_info(f"Will use these hyperparameters: {best_params}")
            return best_params
        else: 
            raise NotImplementedError(f"{self.method} has not been implemented. ")
        
    def create_hyperparameter_widgets(self):
        widgets_list = []

        # General Parameters
        title = widgets.HTML(f"<h1>Hyperparameter Sweep</h1>")
        info = widgets.HTML(f"<p>During hyperparameter sweeps it can be nessecary to reduce the dataset size or the number of epochs trained due to computational cost. However, this will influende the accuracy of the hyperparameter search. The DL specialist chose a default, which you can change in the following. Increase the slider to get a more accurate hyperparameter search (but more computational cost/longer runtime), or decrease the slider for a quicker but less accurate sweep. </p>")
        
        
        train_subset_slider = widgets.IntSlider(
            value=int(self.config['train_subset'] * 100),
            min=0,
            max=100,
            step=1,
            description="Train Subset [%]:",
            style={'description_width': 'initial'}
        )

        reduce_epochs_slider = widgets.IntSlider(
            value=int(self.config['reduce_epochs'] * 100),
            min=0,
            max=100,
            step=1,
            description="Reduce Epochs [%]:",
            style={'description_width': 'initial'}
        )

        # Labels for displaying computed values
        dataset_size_label = widgets.HTML()
        epochs_label = widgets.HTML()
        subset_size = max([1,int(len(self.model_trainer.train_loader.dataset) * (train_subset_slider.value / 100))])
        num_epochs = max([1,int(self.model_trainer.num_epochs * (reduce_epochs_slider.value / 100))])
        
        dataset_size_label.value = f"<b>Resulting Dataset Size:</b> {subset_size} samples"
        epochs_label.value = f"<b>Resulting Epochs:</b> {num_epochs}"
        
        train_subset = widgets.HBox([train_subset_slider, dataset_size_label])
        reduce_epochs = widgets.HBox([reduce_epochs_slider, epochs_label])

        # Function to update labels dynamically
        def update_labels(*args):
            subset_size = max([1,int(len(self.model_trainer.train_loader.dataset) * (train_subset_slider.value / 100))])
            num_epochs = max([1,int(self.model_trainer.num_epochs * (reduce_epochs_slider.value / 100))])
            
            dataset_size_label.value = f"<b>Resulting Dataset Size:</b> {subset_size} samples"
            epochs_label.value = f"<b>Resulting Epochs:</b> {num_epochs}"

        # Attach update function to slider changes
        train_subset_slider.observe(update_labels, names='value')
        reduce_epochs_slider.observe(update_labels, names='value')
        
        method = widgets.HTML(f"<b>Method:</b> {self.config['method']}")
        
        parameter_str = f"<hr><h2>Fixed Parameters</h2><p>A set of parameters which are predefined by the DL expert, and will not be tuned during hyperparameter tuning.</p>"
        for k in self.config['parameter'].keys():
            parameter_str += f"<p> <b>{k}:</b> {self.config['parameter'][k]['value']}</p>"
            parameter_str += f"<p style='font-size:80%'> {self.config['parameter'][k]['explanation']}</p>"
            
        parameter = widgets.HTML(parameter_str)
        

        widgets_list.extend([title, info, train_subset, reduce_epochs, method, parameter, widgets.HTML("<hr><h1>Hyperparameters</h1><p>The DL specialist has set default sweep parameters, however if you wish to change them, you can do so below. Each hyperparameter should be separated by a comma (,) to define a list of sweepable parameters.</p>")])

        # Hyperparameters
        for param, details in self.config['hyperparameter'].items():
            explanation = widgets.HTML(f"<h2>{param}</h2> <p>{details['explanation']}</p>")
            default = widgets.HTML(f"<b>Default:</b> {details['default']}")
            # values = widgets.Text(value=str(details.get('values', 'N/A')), description='Values:')
            values = widgets.Text(value=", ".join([str(v) for v in details['values']]), description='Values:')#str(details.get('values', 'N/A')), description='Values:')
            

            widgets_list.extend([explanation, default, values, widgets.HTML("<br>")])

        return widgets.VBox(widgets_list)