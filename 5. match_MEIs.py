import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
from sensorium.utility.training import read_config
from sensorium.utility import submission
from nnfabrik.builder import get_data, get_model
from sensorium.models.ensemble import EnsemblePrediction
from sensorium.utility import get_correlations
from sensorium.utility.measure_helpers import get_df_for_scores
from loaddata.session_info import load_sessions
from utils.plotting_style import *  # get all the fixed color schemes
from utils.imagelib import load_natural_images
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.rf_lib import *
import pickle as pkl

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

print('Working directory:', os.getcwd())

run_config = read_config('run_config.yaml') # Must be set

RUN_NAME = run_config['current_vals']['RUN_NAME'] # MUST be set. Creates a subfolder in the runs folder with this name, containing data, saved models, etc. IMPORTANT: all values in this folder WILL be deleted.
RUN_FOLDER = run_config['current_vals']['RUN_FOLDER']
area_of_interest = run_config['current_vals']['data']['area_of_interest']
INPUT_FOLDER = run_config['data']['INPUT_FOLDER']
sessions_to_keep = run_config['data']['sessions_to_keep']
num_models = run_config['dev']['num_models']
validate_MEIs = run_config['MEIs']['validate_MEIs']
session_reference = run_config['MEIs']['validation_session']
validation_session_input_folder = run_config['MEIs']['validation_session_input_folder']
session_reference = [[run_config['MEIs']['session_id'], run_config['MEIs']['session_date']]]

print(f'Starting evaluation for {RUN_NAME} with area of interest {area_of_interest}')

if not validate_MEIs:
    print("Flag to validate MEIs not set. Exiting.")
else:
    warnings.filterwarnings('ignore')
    # ### Load configuration for model

    # Loading config only for ensemble 0, because all 5 models have the same config (except
    # for the seed and dataloader train/validation split)

    config_file = f'{RUN_FOLDER}/config_m4_ens0/config.yaml'
    config = read_config(config_file)
    config['model_config']['data_path'] = f'{RUN_FOLDER}/data'
    print(config)
    # ### Prepare dataloader

    # Use only one dataloader, since test and final_test are the same for all ensembles
    # basepath = "notebooks/data/"
    # filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]
    # filenames = [file for file in filenames if 'static26872-17-20' not in file]

    basepath = f'{RUN_FOLDER}/data'
    # Add Add folders two levels deep from basepath into a list
    # First level
    folders = [os.path.join(basepath, name) for name in os.listdir(
        basepath) if os.path.isdir(os.path.join(basepath, name)) and not "merged_data" in name]
    # Second level
    folders = [os.path.join(folder, name) for folder in folders for name in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, name)) and not "merged_data" in name]
    folders = [x.replace("\\", "/") for x in folders]
    print(folders)

    try: 
        session_folders
    except NameError:
        # First level
        session_folders = [os.path.join(INPUT_FOLDER, name) for name in os.listdir(
            INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, name)) and not "merged_data" in name]
        session_folders = [x.replace("\\", "/") for x in session_folders]
        # Second level
        files = [[session_folder, os.path.join(session_folder, name).replace('\\', '/')] for session_folder in session_folders for name in os.listdir(
            session_folder) if os.path.isdir(os.path.join(session_folder, name)) and not "merged_data" in name]
        # only get last value after /
        session_list = [[session_folder.split("/")[-1], name.split("/")[-1]]
                        for session_folder, name in files]

        # drop ['LPE10919', '2023_11_08'] because the data is not converted yet
        session_list = [x for x in session_list if x != ['LPE10919', '2023_11_08']]

    if sessions_to_keep != 'all':
        session_list = [x for x in session_list if x in sessions_to_keep]

    session_list = np.array(session_list)

    print(session_list)

    sessions, nSessions = load_sessions(protocol='IM', session_list=session_list, data_folder = INPUT_FOLDER)

    for ises in range(nSessions):    # Load proper data and compute average trial responses:
        sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

    sessions = compute_pairwise_anatomical_distance(sessions)
    sessions = smooth_rf(sessions,radius=75,rf_type='Fneu')
    sessions = exclude_outlier_rf(sessions) 
    sessions = replace_smooth_with_Fsig(sessions) 

    