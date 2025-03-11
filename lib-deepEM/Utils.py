import ipywidgets as widgets
import json


def format_time(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    time_str = f"{int(days)}d{int(hours)}h{int(minutes)}m{int(seconds)}s" if days > 0 else f"{int(hours)}h{int(minutes)}m{int(seconds)}s"
    return time_str

def print_info(text):
    print("[INFO]::"+text)
    
def print_error(text):
    print("[INFO]::"+text)
    
def print_warning(text):
    print("[INFO]::"+text)

   
def create_text_widget(name,value,description):
    text_widget = widgets.Text(
        value=str(value),
        description=name,
        style={'description_width': 'initial'}
    )
    description_widget = widgets.HTML(value=f"<b>Hint:</b> {description}")
        
    return (text_widget, description_widget)


def load_json(file):
    # Open and load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def extract_defaults(config):
    defaults = {}
    for key, value in config.items():
        if isinstance(value, dict) and "default" in value:
            # Extract default value from nested hyperparameter dict
            defaults[key] = value["default"]
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            nested_defaults = extract_defaults(value)
            defaults.update(nested_defaults)
        else:
            pass
    return defaults

def get_fixed_parameters(config_file):
    params_json = load_json(config_file)["parameter"]
    fixed_parameter = {}
    for k in params_json.keys(): 
        fixed_parameter[k] = params_json[k]["value"]
        
    return fixed_parameter
        




