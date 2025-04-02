import ipywidgets as widgets
import json
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    """Set all relevant seeds for reproducibility in PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, ensures deterministic behavior in PyTorch.
        benchmark (bool): If True, enables CUDNN benchmark mode (faster, but non-deterministic).
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch CUDA
    torch.cuda.manual_seed_all(seed)  # Multi-GPU safe

    # Ensuring deterministic behavior in PyTorch (if required)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark  # Benchmark mode can be disabled for full determinism




def find_file(root_dir: str, filename: str) -> str:
    """
    Recursively searches for a file within a directory.

    Args:
        root_dir (str): The root directory to search in.
        filename (str): The name of the file to find.

    Returns:
        str: The absolute path to the file if found, else None.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)  # Return the first match
    return None  # If the file is not found


def find_model_file(input_path: str) -> str:
    """
    Finds the model checkpoint file (`best_model.pth`) in the given path.

    If `input_path` is a file, checks if it's named "best_model.pth". 
    If `input_path` is a directory, searches recursively for "best_model.pth".

    Args:
        input_path (str): Path to a model file or directory containing it.

    Returns:
        str: Absolute path to the model checkpoint if found, else None.
    """
    if os.path.isfile(input_path):
        if os.path.basename(input_path) == "best_model.pth":
            print_info(f"Found model checkpoint at {input_path}")
            return input_path
        elif input_path.lower().endswith(('.pth', '.pt')):
            print_error("Provided file is no .pth or .pt file.")
            return None
        else:
            print_warning("Provided file is not named 'best_model.pth'. Expected 'best_model.pth'.")
            return input_path
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            if "best_model.pth" in files:
                model_file = os.path.join(root, "best_model.pth")
                if "TrainingRun" in model_file:
                    print_info(f"Found model checkpoint at {model_file}")
                    return model_file
        print_error("No 'best_model.pth' was found for a TrainingRun within the provided directory.")
        return None
    else:
        print_error("Invalid model path: not a file or directory.")
        return None


def format_time(seconds: int) -> str:
    """
    Converts a duration in seconds into a human-readable format.

    Args:
        seconds (int): Duration in seconds.

    Returns:
        str: Formatted time string (e.g., "1h30m15s").
    """
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(days)}d{int(hours)}h{int(minutes)}m{int(seconds)}s" if days > 0 else f"{int(hours)}h{int(minutes)}m{int(seconds)}s"


def print_info(text: str) -> None:
    """
    Prints an informational message.

    Args:
        text (str): The message to print.
    """
    print("[INFO]::" + text)


def print_error(text: str) -> None:
    """
    Prints an error message.

    Args:
        text (str): The message to print.
    """
    print("[ERROR]::" + text)


def print_warning(text: str) -> None:
    """
    Prints a warning message.

    Args:
        text (str): The message to print.
    """
    print("[WARNING]::" + text)


def create_text_widget(name: str, value: str, description: str):
    """
    Creates an interactive text input widget with a description.

    Args:
        name (str): Label for the widget.
        value (str): Default value of the text input.
        description (str): Hint text displayed below the widget.

    Returns:
        tuple: A tuple containing the text widget and its description widget.
    """
    text_widget = widgets.Text(
        value=str(value),
        description=name,
        style={'description_width': 'initial'}, 
        layout={'width': '1000px'}
    )
    description_widget = widgets.HTML(value=f"<b>Hint:</b> {description}")

    return text_widget, description_widget


def create_checkbox_widget(name: str, value: bool, description: str):
    """
    Creates an interactive checkbox widget with a description.

    Args:
        name (str): Label for the widget.
        value (bool): Default state of the checkbox (True for checked, False for unchecked).
        description (str): Hint text displayed below the widget.

    Returns:
        tuple: A tuple containing the checkbox widget and its description widget.
    """
    checkbox_widget = widgets.Checkbox(
        value=value,
        description=name,
        style={'description_width': 'initial'}
    )
    description_widget = widgets.HTML(value=f"<b>Hint:</b> {description}")

    return checkbox_widget, description_widget



def load_json(file: str) -> dict:
    """
    Loads a JSON file.

    Args:
        file (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content as a dictionary.
    """
    with open(file, 'r') as f:
        return json.load(f)


def extract_defaults(config: dict) -> dict:
    """
    Extracts default values from a nested configuration dictionary.

    Args:
        config (dict): The configuration dictionary containing hyperparameters.

    Returns:
        dict: A dictionary containing default values for hyperparameters.
    """
    defaults = {}
    for key, value in config.items():
        if isinstance(value, dict) and "default" in value:
            # Extract default value from nested hyperparameter dict
            defaults[key] = value["default"]
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            nested_defaults = extract_defaults(value)
            defaults.update(nested_defaults)
    return defaults


def get_fixed_parameters(config_file: str) -> dict:
    """
    Extracts fixed parameter values from a JSON configuration file.

    Args:
        config_file (str): Path to the configuration JSON file.

    Returns:
        dict: Dictionary containing fixed parameter names and their values.
    """
    params_json = load_json(config_file)["parameter"]
    fixed_parameter = {k: params_json[k]["value"] for k in params_json.keys()}
    return fixed_parameter