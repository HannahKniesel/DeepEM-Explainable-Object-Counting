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
            title = widgets.HTML(f"<h2>Best hyperparameters</h2><p>Found best hyperparameters (val_loss = {hyperparameters['val_loss']:.4f}) for current dataset ({Path(self.logger.data_path).stem}).")
        else: 
            title = widgets.HTML(f"<h2>Warning</h2><p>Could not find best hyperparameters for current dataset ({Path(self.logger.data_path).stem}). Make sure to conduct a hyperparameter sweep for the best possible performance.")
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
             
        
        
    # Interactive Widgets for JSON Input
    def create_hyperparameter_widgets(self):
        widgets_list = []

        # General Parameters
        title = widgets.HTML(f"<h1>Hyperparameter Sweep</h1> \
                             <p>A hyperparameter sweep is a systematic exploration of different hyperparameter configurations to optimize the performance of a machine learning model. Hyperparameters are settings that are not learned during training but are specified prior to the training process, such as learning rate, batch size, or the number of layers in a neural network.</p> \
                             <p>The DL specialist prepared common parameters which require tuning in order for the DL model to perform well. \
                             While he predefined well suited ranges for tuning, you are able to adapt these ranges in the following.</p> \
                             <p>If you wish to train a model without parameter sweeps (not recommended!) the provided default values will be used.</p>")
        info = widgets.HTML(f"<p>During hyperparameter sweeps it can be nessecary to reduce the dataset size or the number of epochs trained due to computational cost. However, this will influende the accuracy of the hyperparameter search. The DL specialist chose a well suited trade off by predefining these as follows: </p>")
        
        train_subset = widgets.HTML(f"<b>Train Subset:</b> {int(self.config['train_subset']*100)}%")
        reduce_epochs = widgets.HTML(f"<b>Reduce Epochs:</b> {int(self.config['reduce_epochs']*100)}%")
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
    
    def update_config(self, widget_box):
        """Update the JSON configuration based on the values in the widget form."""
        children = widget_box.children
        index = 7  # Starting index for hyperparameters (after general parameters and <hr>)

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
        best_params, best_loss = None, float("inf")

        total_combinations = prod(len(v) for v in search_space.values())
        
        accum_time = 0
        for index,params in enumerate(product(*search_space.values())):
            self.logger.init(f"Sweep_{index}")
            self.logger.log_info(f"Start Sweep {index+1} of {total_combinations}...")
            
            hyperparams = dict(zip(search_space.keys(), params))
            
            try:
                self.logger.log_info(f"Current hyperparams {hyperparams}")
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
                
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = hyperparams
                best_params["val_loss"] = val_loss
                best_index = index
                self.logger.log_best_sweepparameters(best_params)
                

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
        
