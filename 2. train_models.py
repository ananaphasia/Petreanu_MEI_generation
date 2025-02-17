import os 
import sys
import subprocess
from sensorium.utility.training import read_config

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

run_config = read_config('run_config.yaml') # Must be set

RUN_NAME = run_config['current_vals']['RUN_NAME'] # MUST be set. Creates a subfolder in the runs folder with this name, containing data, saved models, etc. IMPORTANT: all values in this folder WILL be deleted.
RUN_FOLDER = run_config['current_vals']['RUN_FOLDER']
num_models = run_config['dev']['num_models']

print(f'Starting training for {RUN_NAME}')

for i in range(num_models):
    subprocess.run(['python', 'scripts/train_model.py', \
                    '-m', f'config_m4_ens{i}', \
                    '-l', RUN_FOLDER, \
                    '-dl', f'{RUN_FOLDER}/data', \
                    '-iwm', str(True) if run_config['model']['init_w_mean_activity'] else str(False)])