import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import os 
from pathlib import Path

from deepEM.Utils import create_text_widget, print_info, print_error, print_warning, find_file, load_json, extract_defaults

# default_datapath = ""#"data/tem-herpes/"
data_link="https://viscom-ulm.github.io/DeepEM/your-contribution.html"


def gen_table(config):

    # Create a list to hold the rows for the table
    rows = []
    input_widgets = {}

    # Loop through hyperparameters in the config and create a row for each
    for idx, (param, details) in enumerate(config['hyperparameter'].items()):
        # Create the value widget, allowing the user to modify the values
        values_widget = widgets.Text(value=", ".join([str(v) for v in details['values']]), description='', layout=widgets.Layout(width="50%", padding="5px" ))
        input_widgets[param] = values_widget
        # Create the explanation widget (non-editable)
        # explanation_widget = widgets.HTML(value=f"<p>{details['explanation']}</p>", layout=widgets.Layout(width="33%", padding="0px 5px", text_align="right"))
        
        # Alternate row background color for readability
        row_background = "#f9f9f9" if idx % 2 == 0 else "#ffffff"
        
        # Create a row for this hyperparameter
        row = widgets.HBox([
            widgets.HTML(value=f"<b>{param}</b> {details['explanation']}</p>", layout=widgets.Layout(width="50%", padding="5px", background_color=row_background, text_align="center")),
            values_widget,
            # explanation_widget
        ])
        
        # Append the row to the list
        rows.append(row)

    # Combine the header and rows into a VBox
    return widgets.VBox(rows), input_widgets



def edit_hyperparameters(data_path):
    hyperparams_path = os.path.join(data_path, "Sweep_Parameters", "best_sweep_parameters.json")
    try: 
        hyperparameters = load_json(hyperparams_path)
    except:
        hyperparameters = None
    
    if(hyperparameters):
        title = widgets.HTML(f"<h2>Best hyperparameters</h2><p>Found best hyperparameters (val_loss = {hyperparameters['val_loss']:.4f}) for current dataset ({Path(data_path).stem}).")
    else: 
        title = widgets.HTML(f"<h2>Warning</h2><p>Could not find best hyperparameters for current dataset ({Path(data_path).stem}). To reach the best possible performance, you should do a hyperparameter sweep.")
        hyperparameters = load_json(os.path.join("configs","parameters.json"))
        hyperparameters = extract_defaults(hyperparameters)
    
    widgets_list = [title]
    for k in hyperparameters.keys():
        if(k != "val_loss"):
            widgets_list.append(widgets.Text(value=str(hyperparameters[k]), description=f'{k}'))
    return widgets.VBox(widgets_list)



# Function to update the config dictionary based on user input
def update_config(config, input_widgets):
    for param, details in config['hyperparameter'].items():
        # Update the config values based on user input
        input_value = input_widgets[param].value
        # Update the config with the new value (convert to appropriate type if needed)
        config['hyperparameter'][param]['values'] = list(map(float, input_value.split(',')))
    return config


def create_hyperparameter_widgets(config):
    
    widgets_list = []

    # General Parameters
    title = widgets.HTML(f"<h3>Custom Sweep</h3><hr><h4>Sweep Settings</h4><p><b>Train Subset:</b> {int(config['train_subset']*100)}%<br><b>Reduce Epochs:</b> {int(config['reduce_epochs']*100)}%<br><b>Method:</b> {config['method']}</p>")
    
    parameter_str = "<hr><h4>Non-Tunable Parameters</h4><table border='1' style='border-collapse: collapse; width: 100%;'>"
    parameter_str += "<tr><th>Parameter</th><th>Value</th><th>Explanation</th></tr>"

    for k in config['parameter'].keys():
        value = config['parameter'][k]['value']
        explanation = config['parameter'][k]['explanation']
        parameter_str += f"<tr><td><b>{k}</b></td><td>{value}</td><td style='font-size: 80%;'>{explanation}</td></tr>"

    parameter_str += "</table>"

        
    parameter = widgets.HTML(parameter_str)
    
    table, input_widgets = gen_table(config)

    widgets_list.extend([title, 
                         widgets.HTML("<hr><h4>Tunable Parameters</h4><p><i>The DL specialist has set default sweep parameters, however if you wish to change them, you can do so below. Each hyperparameter should be separated by a comma (,) to define a list of sweepable parameters.</i></p>"),
                         table,
                         parameter, 
                         ])




    return widgets.VBox(widgets_list), input_widgets
