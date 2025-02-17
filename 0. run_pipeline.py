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

for i, area in enumerate(areas):

    run_config['current_vals']['RUN_NAME'] = f'{area}{"_" if area is not None else ""}{run_config["RUN_NAME"]}'

    RUN_FOLDER_OVERWRITE = run_config['RUN_FOLDER_OVERWRITE']
    if RUN_FOLDER_OVERWRITE == 'None':
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

    yaml_config = yaml.YAML()
    with open('run_config.yaml', 'w') as file:
        yaml_config.dump(run_config, file)

    raise Exception

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