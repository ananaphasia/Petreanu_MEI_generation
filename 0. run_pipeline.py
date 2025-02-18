import subprocess
import os
import sys
from sensorium.utility.training import read_config
import ruamel.yaml as yaml

# Set working directory to root of repo
current_path = os.getcwd()
# Identify if path has 'molanalysis' as a folder in it
if 'Petreanu_MEI_generation' in current_path:
    # If so, set the path to the root of the repo
    current_path = current_path.split('Petreanu_MEI_generation')[0] + 'Petreanu_MEI_generation'
else:
    raise FileNotFoundError(
        f'This needs to be run somewhere from within the Petreanu_MEI_generation folder, not {current_path}')
os.chdir(current_path)
sys.path.append(current_path)

# list of relative locations, in order, of the python scripts to be run
files_list = [
    'IM_dataconversion.py',
    "1. preprocess_data.py",
    "2. train_models.py",
    "3. evaluate.py",
    "4. generateMEIs.py"
    ]

current_path = os.getcwd()

run_config = read_config('run_config.yaml') # Must be set
areas = run_config['data']['areas_of_interest']

class NullRepresenter(yaml.representer.SafeRepresenter):
    def represent_none(self, value):
        return self.represent_scalar('tag:yaml.org,2002:null', 'null')  # Represent None as 'null'

# Custom representer for True/False (with proper casing)
class BoolRepresenter(yaml.representer.SafeRepresenter):
    def represent_bool(self, value):
        return self.represent_scalar('tag:yaml.org,2002:bool', str(value))  # Convert True/False to 'True'/'False'
    
yaml_config = yaml.YAML()
yaml_config.representer.add_representer(type(None), NullRepresenter.represent_none)
yaml_config.representer.add_representer(bool, BoolRepresenter.represent_bool)

if isinstance(run_config['RUN_NAME_S'], str):
    run_names = [run_config['RUN_NAME_S']]
elif isinstance(run_config['RUN_NAME_S'], list):
    run_names = run_config['RUN_NAME_S']
else:
    raise ValueError(f'RUN_NAME_S must be set as a string or a list, not {type(run_config["RUN_NAME_S"])}')

for run_idx, run_name in enumerate(run_names):
    for i, area in enumerate(areas):

        # Update run_config.yaml based on which area is being investigated

        run_config['current_vals']['RUN_NAME'] = f'{area}{"_" if area is not None else ""}{run_name}'

        RUN_FOLDER_OVERWRITE = run_config['RUN_FOLDER_OVERWRITE']
        if RUN_FOLDER_OVERWRITE == 'None' or RUN_FOLDER_OVERWRITE is None:
            RUN_FOLDER = f"runs/{run_config['current_vals']['RUN_NAME']}"
        elif isinstance(RUN_FOLDER_OVERWRITE, str):
            RUN_FOLDER = f'{RUN_FOLDER_OVERWRITE}_{area}'
        elif isinstance(RUN_FOLDER_OVERWRITE, list):
            RUN_FOLDER = RUN_FOLDER_OVERWRITE[i]
        else:
            raise ValueError(f'RUN_FOLDER_OVERWRITE must be None, str, or list, not {type(RUN_FOLDER_OVERWRITE)}')


        run_config['current_vals']['RUN_FOLDER'] = RUN_FOLDER
        run_config['current_vals']['area_id'] = i
        run_config['current_vals']['data']['area_of_interest'] = area

        with open('run_config.yaml', 'w') as file:
            yaml_config.dump(run_config, file)

        # Update each model's configs to set values for V1 and PM max jitter

        for m in range(run_config['dev']['num_models']):
            config_file = os.path.join('model_configs', f'config_m4_ens{m}.yaml')
            config = read_config(config_file)

            # if area == 'V1':
            #     # config['model_config']['max_jitter'] = 0.052
            #     config['model_config']['max_jitter'] = 0
            # elif area == 'PM':
            #     # config['model_config']['max_jitter'] = 0.111
            #     config['model_config']['max_jitter'] = 0
            # else:
            #     print(f'WARNING: Setting max_jitter to that of PM, 0, since current area {area} is not V1 or PM')
            #     # config['model_config']['max_jitter'] = 0.111
            #     config['model_config']['max_jitter'] = 0

            if run_idx == 0: # with grid mean predictor
                if area == 'V1':
                    config['model_config']['max_jitter'] = 0.052
                    # config['model_config']['max_jitter'] = 0
                elif area == 'PM':
                    config['model_config']['max_jitter'] = 0.111
                    # config['model_config']['max_jitter'] = 0
                else:
                    print(f'WARNING: Setting max_jitter to that of PM, 0.111, since current area {area} is not V1 or PM')
                    config['model_config']['max_jitter'] = 0.111
                    # config['model_config']['max_jitter'] = 0

                config['model_config']['grid_mean_predictor'] = {'type': 'cortex',
                            'input_dimensions': 2,
                            'hidden_layers': 4,
                            'hidden_features': 20,
                            'nonlinearity': 'ReLU',
                            'final_tanh': True}
            elif run_idx == 1: # without grid mean predictor
                if area == 'V1':
                    # config['model_config']['max_jitter'] = 0.052
                    config['model_config']['max_jitter'] = 0
                elif area == 'PM':
                    # config['model_config']['max_jitter'] = 0.111
                    config['model_config']['max_jitter'] = 0
                else:
                    print(f'WARNING: Setting max_jitter to that of PM, 0, since current area {area} is not V1 or PM')
                    # config['model_config']['max_jitter'] = 0.111
                    config['model_config']['max_jitter'] = 0

                config['model_config']['grid_mean_predictor'] = None

            with open(config_file, 'w') as file:
                yaml_config.dump(config, file)

        # Run each script in the list
        for file in files_list:
            print(f'Running {file}')
            if '../' in file:
                os.chdir(os.path.join('..', file.split('/')[1]))
                print(f'Changed directory to {os.getcwd()}')
            else:
                os.chdir(current_path)
                print(f'Changed directory to {os.getcwd()}')
            subprocess.run(['python', file])