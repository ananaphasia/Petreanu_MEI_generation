# # Run ensemble model and submit predictions
# ### Imports

import sys
import os
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

from sensorium.utility.training import read_config

run_config = read_config('run_config.yaml') # Must be set

RUN_NAME = run_config['RUN_NAME'] # MUST be set. Creates a subfolder in the runs folder with this name, containing data, saved models, etc. IMPORTANT: all values in this folder WILL be deleted.
area_of_interest = run_config['data']['area_of_interest']
INPUT_FOLDER = run_config['data']['INPUT_FOLDER']
OUT_NAME = f'runs/{RUN_NAME}'

print(f'Starting evaluation for {RUN_NAME} with area of interest {area_of_interest}')

from sensorium.utility.training import read_config
from sensorium.utility import submission
from nnfabrik.builder import get_data, get_model, get_trainer
import torch

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

import seaborn as sns
from tqdm.auto import tqdm


import warnings

warnings.filterwarnings('ignore')
# ### Load configuration for model

# Loading config only for ensemble 0, because all 5 models have the same config (except
# for the seed and dataloader train/validation split)

config_file = 'config_m4_ens0.yaml'
config = read_config(config_file)
config['model_config']['data_path'] = f'{OUT_NAME}/data'
print(config)
# ### Prepare dataloader

# Use only one dataloader, since test and final_test are the same for all ensembles
# basepath = "notebooks/data/"
# filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]
# filenames = [file for file in filenames if 'static26872-17-20' not in file]

basepath = f'{OUT_NAME}/data'
# Add Add folders two levels deep from basepath into a list
# First level
folders = [os.path.join(basepath, name) for name in os.listdir(
    basepath) if os.path.isdir(os.path.join(basepath, name)) and not "merged_data" in name]
# Second level
folders = [os.path.join(folder, name) for folder in folders for name in os.listdir(
    folder) if os.path.isdir(os.path.join(folder, name)) and not "merged_data" in name]
folders = [x.replace("\\", "/") for x in folders]
print(folders)

dataset_fn = config['dataset_fn']  # 'sensorium.datasets.static_loaders'
dataset_config = {'paths': folders,  # filenames,
                  **config['dataset_config'],
                  }

dataloaders = get_data(dataset_fn, dataset_config)

# ### Load trained models

# Instantiate all five models
model_list = list()

for i in tqdm(range(5)):
    # all models have the same parameters
    # e.g. 'sensorium.models.modulated_stacked_core_full_gauss_readout'
    model_fn = config['model_fn']
    model_config = config['model_config']

    model = get_model(model_fn=model_fn,
                      model_config=model_config,
                      dataloaders=dataloaders,
                      seed=config['model_seed'],
                      )

    # Load trained weights from specific ensemble
    # save_file = 'saved_models/config_m4_ens{}/saved_model_v1.pth'.format(i)
    save_file = f'{OUT_NAME}/config_m4_ens{i}/saved_model_v1.pth'
    model.load_state_dict(torch.load(save_file))
    model_list.append(model)

# ### Combine them into one ensemble model

from sensorium.models.ensemble import EnsemblePrediction

ensemble = EnsemblePrediction(model_list, mode='mean')

type(model_list[0])

# ### Generate submission file

# dataset_name = '27204-5-13'

# submission.generate_submission_file(trained_model=ensemble, 
#                                     dataloaders=dataloaders,
#                                     data_key=dataset_name,
#                                     path="notebooks/submission_m4/results/",
#                                     device="cuda")

# ### Evaluate model on all datasets

from sensorium.utility import get_correlations, get_signal_correlations, get_fev
from sensorium.utility.measure_helpers import get_df_for_scores

# #### Test data

tier = "validation"

single_trial_correlation = get_correlations(
    ensemble, dataloaders, tier=tier, device="cuda", as_dict=True)

df = get_df_for_scores(session_dict=single_trial_correlation,
                       measure_attribute="Single Trial Correlation"
                       )

for k in dataloaders[tier]:
    assert len(df[df['dataset'] == k]) == len(dataloaders[tier][k].dataset.neurons.area), f"Length of df and dataloader not equal, {len(df[df['dataset'] == k])} != {len(dataloaders[tier][k].dataset.neurons.area)}"
    df.loc[df['dataset'] == k, 'area'] = dataloaders[tier][k].dataset.neurons.area

# data_basepath = "../molanalysis/data/IM/"
data_basepath = f'{INPUT_FOLDER}/'
# respmat_data_basepath = f'../molanalysis/MEI_generation/data/{RUN_NAME}'
respmat_data_basepath = f'{OUT_NAME}/data'

for k in dataloaders[tier]:
    data_path = os.path.join(data_basepath, k.split('-')[1].split('_')[0] + '/' + '_'.join(k.split('-')[1].split('_')[1:]))
    celldata = pd.read_csv(data_path + '/celldata.csv')
    celldata = celldata.loc[celldata['roi_name'] == area_of_interest] if area_of_interest is not None else celldata
    assert len(df[df['dataset'] == k]) == len(celldata), f"Length of df and celldata not equal, {len(df[df['dataset'] == k])} != {len(celldata)} of {k}"
    df.loc[df['dataset'] == k, 'labeled'] = celldata['redcell'].astype(bool).values
    df.loc[df['dataset'] == k, 'cell_id'] = celldata['cell_id'].values

    trialdata = pd.read_csv(data_path + '/trialdata.csv')
    if 'repetition' not in trialdata:
        trialdata['repetition'] = np.empty(np.shape(trialdata)[0])
        for iT in range(len(trialdata)):
            trialdata.loc[iT,'repetition'] = np.sum(trialdata['ImageNumber'][:iT] == trialdata['ImageNumber'][iT])

    nNeurons = len(celldata)

    respmat_data_path = os.path.join(respmat_data_basepath, k.split('-')[1].split('_')[0] + '/' + '_'.join(k.split('-')[1].split('_')[1:]), 'data')
    respmat = np.load(respmat_data_path + '/respmat.npy')
    respmat = respmat[celldata.index.values]
    
    # Compute the covariance between the first and the second presentation of each image
    cov_signal = np.zeros(nNeurons)
    for iN in tqdm(range(nNeurons)):
        resp1 = respmat[iN,trialdata['ImageNumber'][trialdata['repetition']==0].index[np.argsort(trialdata['ImageNumber'][trialdata['repetition']==0])]]
        resp2 = respmat[iN,trialdata['ImageNumber'][trialdata['repetition']==1].index[np.argsort(trialdata['ImageNumber'][trialdata['repetition']==1])]]
        cov_signal[iN] = np.cov(resp1,resp2)[0,1]
        
    cov_noise = np.var(respmat,axis=1) - cov_signal
    SNR = cov_signal / cov_noise

plt.hist(SNR, bins=np.arange(-0.1,1.5,0.05))
# plt.show()
os.makedirs(f'{OUT_NAME}/results', exist_ok=True)
plt.savefig(f'{OUT_NAME}/results/SNR_hist.png')

# plt.scatter(SNR, df['Single Trial Correlation'], alpha=0.5, s=3)
# plt.xlabel('SNR')
# plt.ylabel('Single Trial Correlation')
# plt.title('SNR vs Single Trial Correlation')
# plt.show()

# # Do pearson correlation
# from scipy.stats import pearsonr

# pearsonr(SNR, df['Single Trial Correlation'])

# # Do spearman correlation
# from scipy.stats import spearmanr

# spearmanr(SNR, df['Single Trial Correlation'])

for i, dataset_name in enumerate(df['dataset'].drop_duplicates().values):
    df.loc[df['dataset'] == dataset_name, 'dataset'] = f'Dataset {i+1:02}'

plt.rcParams.update({'font.size': 32})
sns.set_theme(font_scale=3.5)

sns.set_context("talk", font_scale=.8)
fig = plt.figure(figsize=(15, 8))
sns.boxenplot(x="dataset", y="Single Trial Correlation", data=df)
plt.xticks(rotation=45)
sns.despine(trim=True)

sns.set_context("talk", font_scale=.8)
fig, axes = plt.subplots(nrows=1, ncols=len(df['dataset'].unique()), figsize=(15, 8), sharey=True)

for idx, (ax, (i, g)) in enumerate(zip(np.array(axes).reshape(-1), df.sort_values('area', ascending = False).groupby('dataset'))):
    sns.boxenplot(x="area", y="Single Trial Correlation", data=g, ax=ax)
    ax.set_title(i)  # Set the title of each subplot to the dataset name
    ax.set_xlabel("")  # Set the x-axis label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 

    if idx > 0:
        # remove y axis line
        ax.spines['left'].set_visible(False)
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.get_yaxis().set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)            

sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{OUT_NAME}/results/area_boxplot.png')

sns.set_context("talk", font_scale=.8)
fig, axes = plt.subplots(nrows=1, ncols=len(df['dataset'].unique()), figsize=(15, 8), sharey=True)

for idx, (ax, (i, g)) in enumerate(zip(np.array(axes).reshape(-1), df.sort_values('labeled', ascending = False).groupby('dataset'))):
    sns.boxenplot(x="labeled", y="Single Trial Correlation", data=g, ax=ax)
    ax.set_title(i)  # Set the title of each subplot to the dataset name
    ax.set_xlabel("")  # Set the x-axis label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 

    if idx > 0:
        # remove y axis line
        ax.spines['left'].set_visible(False)
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.get_yaxis().set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)            

sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{OUT_NAME}/results/labeled_boxplot.png')

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("talk", font_scale=.8)

# Create a FacetGrid to split the data by 'dataset' and 'labeled'
g = sns.FacetGrid(df, col="dataset", row="labeled", margin_titles=True, height=4, aspect=1.5, sharey=True)

# Use boxenplot in each facet
g.map(sns.boxenplot, "area", "Single Trial Correlation")

# Adjust labels and titles
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("", "Single Trial Correlation")

# Rotate x-tick labels for better readability
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

# Adjust layout and remove extra spines
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{OUT_NAME}/results/area_labeled_boxplot.png')


sns.set_context("talk", font_scale=.8)
fig = plt.figure(figsize=(15, 8))
sns.barplot(x="dataset", y="Single Trial Correlation", data=df, )
plt.xticks(rotation=45)
plt.ylim(0.3, 0.5)
sns.despine(trim=True)

fig, axes = plt.subplots(nrows=1, ncols=len(df['dataset'].unique()), figsize=(15, 8), sharey=True)

for ax, (i, g) in zip(np.array(axes).reshape(-1), df.sort_values("area", ascending=False).groupby('dataset')):
    sns.barplot(x="area", y="Single Trial Correlation", data=g, ax=ax)
    ax.set_title(i)  # Set the title of each subplot to the dataset name
    ax.set_xlabel("")  # Set the x-axis label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{OUT_NAME}/results/area_barplot.png')

fig, axes = plt.subplots(nrows=1, ncols=len(df['dataset'].unique()), figsize=(15, 8), sharey=True)

for ax, (i, g) in zip(np.array(axes).reshape(-1), df.sort_values("labeled", ascending=False).groupby('dataset')):
    sns.barplot(x="labeled", y="Single Trial Correlation", data=g, ax=ax)
    ax.set_title(i)  # Set the title of each subplot to the dataset name
    ax.set_xlabel("")  # Set the x-axis label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{OUT_NAME}/results/labeled_barplot.png')

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("talk", font_scale=.8)

# Create a FacetGrid to split the data by 'dataset' and 'labeled'
g = sns.FacetGrid(df, col="dataset", row="labeled", margin_titles=True, height=4, aspect=1.5, sharey=True)

# Use boxenplot in each facet
g.map(sns.barplot, "area", "Single Trial Correlation")

# Adjust labels and titles
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("", "Single Trial Correlation")

# Rotate x-tick labels for better readability
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

# Adjust layout and remove extra spines
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{OUT_NAME}/results/area_labeled_barplot.png')


df_desc = df.groupby('dataset').describe()
df_desc.loc[("All datasets", )] = df_desc.mean()
# I'm so sorry about this horrible one liner
df_desc.loc[("All datasets, weighted"), ] = df_desc['Single Trial Correlation'].mul((df_desc['Single Trial Correlation']['count'].values.reshape(-1, 1)) / np.sum(df_desc['Single Trial Correlation']['count'].values)).sum().values
# df_desc.to_csv('notebooks/submission_m4/results/validation_pred_description.csv', index = False)
df_desc.to_csv(f'{OUT_NAME}/results/validation_pred_description.csv', index=False)
# df_desc

# get index in folders for LPE10885/2023_10_20 if it exists
true_idx = None
for idx, folder in enumerate(folders):
    if 'LPE10885/2023_10_20' in folder:
        true_idx = idx
        break

num_neurons = df_desc['Single Trial Correlation']['count'].iloc[true_idx].astype(int)

mus = np.zeros((5, num_neurons, 2))
sigmas = np.zeros((5, num_neurons, 2, 2))

# model.readout._modules

if true_idx:
    df_trunc = df.loc[df['dataset'] == 'LPE10885-LPE10885_2023_10_20-0']
    for i, model in enumerate(model_list):
        mus[i] = model.readout._modules['LPE10885-LPE10885_2023_10_20-0'].mu.detach().cpu().numpy().reshape(-1, 2)
        sigmas[i] = model.readout._modules['LPE10885-LPE10885_2023_10_20-0'].sigma.detach().cpu().numpy().reshape(-1, 2, 2)

    # raise NotImplementedError("Save as np arrays instead of CSV")

    df_neuron_stats = pd.DataFrame(columns=['dataset', 'neuron', 'mean', 'cov', 'mean_std', 'cov_std'] + [f'mean_{i}' for i in range(5)] + [f'cov_{i}' for i in range(5)])

    df_neuron_stats['dataset'] = df_trunc['dataset']
    df_neuron_stats['neuron'] = np.repeat(np.arange(num_neurons), len(df_trunc['dataset'].unique()))
    for i in range(5):
        df_neuron_stats[f'mean_{i}'] = list(mus[i].round(2))
        df_neuron_stats[f'cov_{i}'] = list(sigmas[i].round(2))

    df_neuron_stats['mean'] = list(mus.mean(axis=0).round(2))
    df_neuron_stats['cov'] = list(sigmas.mean(axis=0).round(2))

    df_neuron_stats['mean_std'] = list(mus.std(axis=0).round(2))
    df_neuron_stats['cov_std'] = list(sigmas.std(axis=0).round(2))

    df_neuron_stats['single_trial_correlation'] = df_trunc['Single Trial Correlation']
    df_neuron_stats['cell_id'] = df_trunc['cell_id']

    # df_neuron_stats.to_csv('notebooks/submission_m4/results/neuron_stats.csv', index = False)
    df_neuron_stats.to_csv(f'{OUT_NAME}/results/neuron_stats.csv', index = False)

else:
    print("LPE10885/2023_10_20 not found in folders")

# df_neuron_stats

# df

# Get fit parameters from the readout, ie x and y coordinates of spatial mask


