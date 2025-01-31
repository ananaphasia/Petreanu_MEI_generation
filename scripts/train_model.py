""" Script to train a sensorium model

Adrian 2022-10-03 """

print('Running script train_model.py')

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil
import warnings
from nnfabrik.builder import get_data, get_model, get_trainer

warnings.filterwarnings('ignore')

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
sys.path.insert(0, '.')  # hacky solution for now, TODO: fix

from sensorium.utility.training import read_config, print_t, set_seed
from sensorium.utility import prediction

# read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', default='model_for_testing')
parser.add_argument('-l', '--save_location', default='saved_models')
parser.add_argument('-cl', '--config_location', default='model_configs')
parser.add_argument('-dl', '--data_location', default='data')
parser.add_argument('-iwm', '--init_w_mean_activity', default=False)
args = parser.parse_args()

model_name = args.model_name
save_location = args.save_location
config_location = args.config_location
data_location = args.data_location
init_w_mean_activity = args.init_w_mean_activity

save_folder = os.path.join( save_location, model_name )
os.makedirs( save_folder, exist_ok=True )

data_folder = ''.join(data_location.split('/data')[:-1])
    
config_file = os.path.join(config_location, model_name+'.yaml' )
shutil.copy( config_file, os.path.join(save_folder, 'config.yaml' ))
config = read_config( config_file )

if config['verbose'] > 0:
    print_t('Loading data for "{}"'.format(model_name))

set_seed( config['model_seed'] )  # seed all random generators

#####################################
## DATALOADER
######################################
if config['data_sets'][0] == 'all':
    # basepath = "notebooks/data/"
    # filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]
    # filenames = [file for file in filenames if 'static26872-17-20' not in file]
    basepath = data_location
    # Add Add folders two levels deep from basepath into a list
    # First level
    folders = [os.path.join(basepath, name) for name in os.listdir(
        basepath) if os.path.isdir(os.path.join(basepath, name)) and not "merged_data" in name]
    # Second level
    folders = [os.path.join(folder, name) for folder in folders for name in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, name)) and not "merged_data" in name]
    folders = [x.replace("\\", "/") for x in folders]
    folders
else:
    filenames = config['data_sets']
    # filenames like ['notebooks/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip', ]

dataset_fn = config['dataset_fn']  # 'sensorium.datasets.static_loaders'
dataset_config = {'paths': folders, # filenames
                  **config['dataset_config'],
                 }

dataloaders = get_data(dataset_fn, dataset_config)

#####################################
## MODEL
######################################
# Instantiate model
model_fn = config['model_fn']     # e.g. 'sensorium.models.modulated_stacked_core_full_gauss_readout'
model_config = config['model_config']

model_config['data_path'] = data_location
model_config['init_w_mean_activity'] = init_w_mean_activity
model_config['mean_activity_basepath'] = data_location

model = get_model(model_fn=model_fn,
                  model_config=model_config,
                  dataloaders=dataloaders,
                  seed=config['model_seed'],
                 )

#####################################
## LOAD PRETRAINED CORE
######################################
if config['use_pretrained_core']:
    save_file = config['pretrained_model_file']
    pretrained_state_dict = torch.load(save_file)
    
    # keep only values of core
    core_only = {k:v for k, v in pretrained_state_dict.items() if 'core.' in k}

    # set only pretrained core values
    ret = model.load_state_dict(core_only, strict=False)


#####################################
## TRAINER
######################################
trainer_fn = config['trainer_fn']   # "sensorium.training.standard_trainer"
trainer_config = config['trainer_config']

trainer = get_trainer(trainer_fn=trainer_fn, 
                     trainer_config=trainer_config)


#####################################
## TRAINING AND EVALUATION
######################################
print_t('Start of model training')
# Train model
validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
print_t('Model training finished')
            
# Save model
torch.save(model.state_dict(), os.path.join(save_folder, 'saved_model_v1.pth') )


# Run predictions

# calculate predictions per dataloader
results = prediction.all_predictions_with_trial(model, dataloaders)

# merge predictions, sort in time and add behavioral variables
merged = prediction.merge_predictions(results)
sorted_res = prediction.sort_predictions_by_time(merged, dataset_sorting_path=f'{data_folder}/dataset_sortings.npy')
prediction.inplace_add_behavior_to_sorted_predictions(sorted_res, data_folder=f'{data_folder}/data')

# calculate correlations on splits
dataframe_entries = list()
trial_trans = { 0:'Train', 1:'Val', 2:'Test', 3:'Final Test'}
keys = list( sorted_res.keys() )

for key in keys:
    # calculate correlations
    ses_data = sorted_res[key]
    nr_neurons = ses_data['output'].shape[1]
    trial_type = ses_data['trial_type']

    for i in range(nr_neurons):
        true = ses_data['target'][:,i]
        pred = ses_data['output'][:,i]

        for split in range(3):
            cor = np.corrcoef( true[ trial_type==split ], pred[ trial_type==split ])[1,0]
            if np.isnan(cor):
                cor=0

            dataframe_entries.append(
                        dict(model=model_name, key=key, neuron=i,
                             split=trial_trans[split], cor=cor)
                        )

df = pd.DataFrame( dataframe_entries )

if config['save_csv']:
    # save DataFrame as csv
    path = os.path.join( save_folder, model_name+'.csv' )
    os.makedirs( os.path.dirname(path), exist_ok=True)
    df.to_csv(path)

if config['save_predictions_npy']:
    # save also the predictions for each neuron
    os.makedirs( save_folder, exist_ok=True)
    np.save( os.path.join(save_folder, model_name+'.npy'), sorted_res)

try:
    print(df.groupby('dataset').describe())
except:
    pass

print_t('Done with evaluation. Exiting...')
