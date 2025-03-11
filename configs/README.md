# Model Training Configuration Documentation

This document describes the configuration format for model training, including fixed and tunable parameters. It provides an overview of each configuration option, its purpose, and how to modify them for different training scenarios.

## Structure Overview

The configuration is organized into different sections:

- `parameter`: Contains fixed parameters for training. The current parameters are required and need to be defined in any case. Additional parameters can be added and accessed via `self.parameter[key]` of the `lib.ModelTrainer.py` module.
- `train_subset`: Defines the proportion of the dataset used for hyperparameter tuning (will use the full dataset for final training).
- `reduce_epochs`: Determines the fraction of epochs for reducing the training time during hyperparameter search.
- `method`: Specifies the method for hyperparameter search (currently only `grid` is supported).
- `hyperparameter`: Defines tunable hyperparameters and their possible values. The current parameters are required and need to be defined in any case. Additional parameters can be added and accessed via `self.parameter[key]` of the `lib.ModelTrainer.py` module. These parameters will be used during a sweep.

### Example Configuration

```json
{
    "parameter": {
        "epochs": {
            "value": 3000,
            "explanation": "An 'epoch' in deep learning is one full pass of the model through all the training data, where the model learns and adjusts to improve its predictions."
        }, 
        "early_stopping_patience": {
            "value": 100,
            "explanation": "'Early stopping patience' is a setting that tells the model to stop training if it doesn't improve after a certain number of tries (validation steps), helping to avoid wasting time on unnecessary training."
        },
        "validation_interval": {
            "value": 100,
            "explanation": "The 'validation interval' is the frequency (in epochs) at which the model checks its performance on a separate set of data during training to see how well it's learning."
        },
        "scheduler_step_by": {
            "value": "epoch", 
            "explanation": "The 'scheduler step by' setting determines whether the optimizer adjusts its learning rate after each 'epoch' (a full pass through the data) or after each 'iteration' (a single step of training on a batch of data)."
        }, 
        "images_to_visualize": {
            "value": 10,
            "explanation": "'Images to visualize' is the number of sample images saved during validation or testing to help evaluate how well the model is making predictions."
        }
    },

    "train_subset": 0.5, 
    "reduce_epochs": 0.1, 
    "method": "grid", 
    "hyperparameter": {
        "learning_rate": {
            "default": 1e-5,
            "values": [1e-6, 1e-5, 1e-4],
            "explanation": "Controls the step size at each iteration while optimizing the model's weights. A higher learning rate may lead to faster convergence but risks overshooting the optimal solution, while a lower rate ensures smoother convergence but may require more training time."
        },
        "batch_size": {
            "default": 16,
            "values": [8,16],
            "explanation": "The number of samples per batch during training. Higher numbers usually lead to more stable training, but eventually lead to 'OOM (out of memory)' errors."
        }
    }
    
    
}
``` 