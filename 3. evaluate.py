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

RUN_NAME = run_config['RUN_NAME'] # MUST be set. Creates a subfolder in the runs folder with this name, containing data, saved models, etc. IMPORTANT: all values in this folder WILL be deleted.
RUN_FOLDER = run_config['RUN_FOLDER_OVERWRITE'] if run_config['RUN_FOLDER_OVERWRITE'] is not None and run_config['RUN_FOLDER_OVERWRITE'] != 'None' else f'runs/{RUN_NAME}'
area_of_interest = run_config['data']['area_of_interest']
INPUT_FOLDER = run_config['data']['INPUT_FOLDER']
num_models = run_config['dev']['num_models']

print(f'Starting evaluation for {RUN_NAME} with area of interest {area_of_interest}')

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

dataset_fn = config['dataset_fn']  # 'sensorium.datasets.static_loaders'
dataset_config = {'paths': folders,  # filenames,
                  **config['dataset_config'],
                  }

dataloaders = get_data(dataset_fn, dataset_config)

# ### Load trained models

# Instantiate all five models
model_list = list()

for i in tqdm(range(num_models)):
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
    save_file = f'{RUN_FOLDER}/config_m4_ens{i}/saved_model_v1.pth'
    model.load_state_dict(torch.load(save_file))
    model_list.append(model)

# ### Combine them into one ensemble model

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

# #### Test data

tier = "validation"

single_trial_correlation = get_correlations(
    ensemble, dataloaders, tier=tier, device="cuda", as_dict=True)

single_trial_correlation_list = []
for i in tqdm(range(num_models), desc="Getting Single Trial Correlations"):
    single_trial_correlation_list.append(get_correlations(
        model_list[i], dataloaders, tier=tier, device="cuda", as_dict=True))

df = get_df_for_scores(session_dict=single_trial_correlation,
                       measure_attribute="Single Trial Correlation"
                       )

df_list = []
for i in tqdm(range(num_models), desc="Getting DF for Scores"):
    df_list.append(get_df_for_scores(session_dict=single_trial_correlation_list[i],
                                     measure_attribute="Single Trial Correlation"
                                     ))

for k in dataloaders[tier]:
    assert len(df[df['dataset'] == k]) == len(dataloaders[tier][k].dataset.neurons.area), f"Length of df and dataloader not equal, {len(df[df['dataset'] == k])} != {len(dataloaders[tier][k].dataset.neurons.area)}"
    df.loc[df['dataset'] == k, 'area'] = dataloaders[tier][k].dataset.neurons.area

for i in range(num_models):
    for k in dataloaders[tier]:
        assert len(df_list[i][df_list[i]['dataset'] == k]) == len(dataloaders[tier][k].dataset.neurons.area), f"Length of df {i} and dataloader not equal, {len(df_list[i][df_list[i]['dataset'] == k])} != {len(dataloaders[tier][k].dataset.neurons.area)}"
        df_list[i].loc[df_list[i]['dataset'] == k, 'area'] = dataloaders[tier][k].dataset.neurons.area

# data_basepath = "../molanalysis/data/IM/"
data_basepath = f'{INPUT_FOLDER}/'
# respmat_data_basepath = f'../molanalysis/MEI_generation/data/{RUN_NAME}'
respmat_data_basepath = f'{RUN_FOLDER}/data'

for k in dataloaders[tier]:
    data_path = os.path.join(data_basepath, k.split('-')[1].split('_')[0] + '/' + '_'.join(k.split('-')[1].split('_')[1:]))
    celldata = pd.read_csv(data_path + '/celldata.csv')
    celldata = celldata.loc[celldata['roi_name'] == area_of_interest] if area_of_interest is not None else celldata
    assert len(df[df['dataset'] == k]) == len(celldata), f"Length of df and celldata not equal, {len(df[df['dataset'] == k])} != {len(celldata)} of {k}"
    df.loc[df['dataset'] == k, 'labeled'] = celldata['redcell'].astype(bool).values
    df.loc[df['dataset'] == k, 'cell_id'] = celldata['cell_id'].values

    for i in range(num_models):
        assert len(df_list[i][df_list[i]['dataset'] == k]) == len(celldata), f"Length of df {i} and celldata not equal, {len(df_list[i][df_list[i]['dataset'] == k])} != {len(celldata)} of {k}"
        df_list[i].loc[df_list[i]['dataset'] == k, 'labeled'] = celldata['redcell'].astype(bool).values
        df_list[i].loc[df_list[i]['dataset'] == k, 'cell_id'] = celldata['cell_id'].values

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
    os.makedirs(f'{RUN_FOLDER}/results', exist_ok=True)
    plt.savefig(f'{RUN_FOLDER}/results/SNR_hist_{k}.png')

# plt.scatter(SNR, df['Single Trial Correlation'], alpha=0.5, s=3)
# plt.xlabel('SNR')
# plt.ylabel('Single Trial Correlation')
# plt.title('SNR vs Single Trial Correlation')
# plt.show()

# # Do pearson correlation
# pearsonr(SNR, df['Single Trial Correlation'])

# # Do spearman correlation
# spearmanr(SNR, df['Single Trial Correlation'])

for i, dataset_name in enumerate(df['dataset'].drop_duplicates().values):
    df.loc[df['dataset'] == dataset_name, 'dataset_name_full'] = dataset_name
    df.loc[df['dataset'] == dataset_name, 'dataset'] = f'Dataset {i+1:02}'


for i in range(num_models):
    for j, dataset_name in enumerate(df_list[i]['dataset'].drop_duplicates().values):
        df_list[i].loc[df_list[i]['dataset'] == dataset_name, 'dataset_name_full'] = dataset_name
        df_list[i].loc[df_list[i]['dataset'] == dataset_name, 'dataset'] = f'Dataset {j+1:02}'

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

    # if idx > 0:
    #     # remove y axis line
    #     ax.spines['left'].set_visible(False)
    #     ax.set_ylabel("")
    #     ax.set_yticklabels([])
    #     ax.set_yticks([])
    #     ax.get_yaxis().set_visible(False)

    #     for spine in ax.spines.values():
    #         spine.set_visible(False)            

plt.suptitle("Single Trial Correlation vs Area")
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{RUN_FOLDER}/results/area_boxplot.png')

for i in range(num_models):
    sns.set_context("talk", font_scale=.8)
    fig, axes = plt.subplots(nrows=1, ncols=len(df_list[i]['dataset'].unique()), figsize=(15, 8), sharey=True)
    for idx, (ax, (j, g)) in enumerate(zip(np.array(axes).reshape(-1), df_list[i].sort_values('area', ascending = False).groupby('dataset'))):
        sns.boxenplot(x="area", y="Single Trial Correlation", data=g, ax=ax)
        ax.set_title(j)  # Set the title of each subplot to the dataset name
        ax.set_xlabel("")  # Set the x-axis label
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 

        # if idx > 0:
        #     # remove y axis line
        #     ax.spines['left'].set_visible(False)
        #     ax.set_ylabel("")
        #     ax.set_yticklabels([])
        #     ax.set_yticks([])
        #     ax.get_yaxis().set_visible(False)

        #     for spine in ax.spines.values():
        #         spine.set_visible(False)            

    plt.suptitle(f"Single Trial Correlation vs Area, Model {i}")
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{RUN_FOLDER}/results/area_boxplot_{i}.png')

sns.set_context("talk", font_scale=.8)
fig, axes = plt.subplots(nrows=1, ncols=len(df['dataset'].unique()), figsize=(15, 8), sharey=True)
for idx, (ax, (i, g)) in enumerate(zip(np.array(axes).reshape(-1), df.sort_values('labeled', ascending = False).groupby('dataset'))):
    sns.boxenplot(x="labeled", y="Single Trial Correlation", data=g, ax=ax)
    ax.set_title(i)  # Set the title of each subplot to the dataset name
    ax.set_xlabel("")  # Set the x-axis label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 

    # if idx > 0:
    #     # remove y axis line
    #     ax.spines['left'].set_visible(False)
    #     ax.set_ylabel("")
    #     ax.set_yticklabels([])
    #     ax.set_yticks([])
    #     ax.get_yaxis().set_visible(False)

    #     for spine in ax.spines.values():
    #         spine.set_visible(False)            
plt.suptitle("Single Trial Correlation vs Labeled")
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{RUN_FOLDER}/results/labeled_boxplot.png')

for i in range(num_models):
    sns.set_context("talk", font_scale=.8)
    fig, axes = plt.subplots(nrows=1, ncols=len(df_list[i]['dataset'].unique()), figsize=(15, 8), sharey=True)
    for idx, (ax, (j, g)) in enumerate(zip(np.array(axes).reshape(-1), df_list[i].sort_values('labeled', ascending = False).groupby('dataset'))):
        sns.boxenplot(x="labeled", y="Single Trial Correlation", data=g, ax=ax)
        ax.set_title(j)  # Set the title of each subplot to the dataset name
        ax.set_xlabel("")  # Set the x-axis label
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 

        # if idx > 0:
        #     # remove y axis line
        #     ax.spines['left'].set_visible(False)
        #     ax.set_ylabel("")
        #     ax.set_yticklabels([])
        #     ax.set_yticks([])
        #     ax.get_yaxis().set_visible(False)

        #     for spine in ax.spines.values():
        #         spine.set_visible(False)            

    plt.suptitle(f"Single Trial Correlation vs Labeled, Model {i}")
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{RUN_FOLDER}/results/labeled_boxplot_{i}.png')

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
plt.suptitle("Single Trial Correlation vs Area, Labeled")
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{RUN_FOLDER}/results/area_labeled_boxplot.png')

for i in range(num_models):
    sns.set_context("talk", font_scale=.8)

    # Create a FacetGrid to split the data by 'dataset' and 'labeled'
    g = sns.FacetGrid(df_list[i], col="dataset", row="labeled", margin_titles=True, height=4, aspect=1.5, sharey=True)

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
    plt.suptitle(f"Single Trial Correlation vs Area, Labeled, Model {i}")
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{RUN_FOLDER}/results/area_labeled_boxplot_{i}.png')


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

plt.suptitle("Single Trial Correlation vs Area")
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{RUN_FOLDER}/results/area_barplot.png')

for i in range(num_models):
    sns.set_context("talk", font_scale=.8)
    fig = plt.figure(figsize=(15, 8))
    sns.barplot(x="dataset", y="Single Trial Correlation", data=df_list[i], )
    plt.xticks(rotation=45)
    plt.ylim(0.3, 0.5)
    sns.despine(trim=True)

    fig, axes = plt.subplots(nrows=1, ncols=len(df_list[i]['dataset'].unique()), figsize=(15, 8), sharey=True)

    for ax, (j, g) in zip(np.array(axes).reshape(-1), df_list[i].sort_values("area", ascending=False).groupby('dataset')):
        sns.barplot(x="area", y="Single Trial Correlation", data=g, ax=ax)
        ax.set_title(j)  # Set the title of each subplot to the dataset name
        ax.set_xlabel("")  # Set the x-axis label
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.suptitle(f"Single Trial Correlation vs Area, Model {i}")
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{RUN_FOLDER}/results/area_barplot_{i}.png')

fig, axes = plt.subplots(nrows=1, ncols=len(df['dataset'].unique()), figsize=(15, 8), sharey=True)

for ax, (i, g) in zip(np.array(axes).reshape(-1), df.sort_values("labeled", ascending=False).groupby('dataset')):
    sns.barplot(x="labeled", y="Single Trial Correlation", data=g, ax=ax)
    ax.set_title(i)  # Set the title of each subplot to the dataset name
    ax.set_xlabel("")  # Set the x-axis label
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.suptitle("Single Trial Correlation vs Labeled")
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{RUN_FOLDER}/results/labeled_barplot.png')


for i in range(num_models):
    sns.set_context("talk", font_scale=.8)
    fig, axes = plt.subplots(nrows=1, ncols=len(df_list[i]['dataset'].unique()), figsize=(15, 8), sharey=True)
    for ax, (j, g) in zip(np.array(axes).reshape(-1), df_list[i].sort_values("labeled", ascending=False).groupby('dataset')):
        sns.barplot(x="labeled", y="Single Trial Correlation", data=g, ax=ax)
        ax.set_title(j)  # Set the title of each subplot to the dataset name
        ax.set_xlabel("")  # Set the x-axis label
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.suptitle(f"Single Trial Correlation vs Labeled, Model {i}")
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{RUN_FOLDER}/results/labeled_barplot_{i}.png')

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
plt.suptitle("Single Trial Correlation vs Area, Labeled")
sns.despine(trim=True)
plt.tight_layout()
# plt.show()
plt.savefig(f'{RUN_FOLDER}/results/area_labeled_barplot.png')

for i in range(num_models):
    sns.set_context("talk", font_scale=.8)

    # Create a FacetGrid to split the data by 'dataset' and 'labeled'
    g = sns.FacetGrid(df_list[i], col="dataset", row="labeled", margin_titles=True, height=4, aspect=1.5, sharey=True)

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
    plt.suptitle(f"Single Trial Correlation vs Area, Labeled, Model {i}")
    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{RUN_FOLDER}/results/area_labeled_barplot_{i}.png')


df_desc = df.groupby('dataset').describe()
df_desc.loc[("All datasets", )] = df_desc.mean()
# I'm so sorry about this horrible one liner
df_desc.loc[("All datasets, weighted"), ] = df_desc['Single Trial Correlation'].mul((df_desc['Single Trial Correlation']['count'].values.reshape(-1, 1)) / np.sum(df_desc['Single Trial Correlation']['count'].values)).sum().values
# df_desc.to_csv('notebooks/submission_m4/results/validation_pred_description.csv', index = False)
df_desc.to_csv(f'{RUN_FOLDER}/results/validation_pred_description.csv', index=False)
# df_desc

df_desc_list = []
for i in range(num_models):
    df_desc_list.append(df_list[i].groupby('dataset').describe())
    df_desc_list[i].loc[("All datasets", )] = df_desc_list[i].mean()
    # I'm so sorry about this horrible one liner
    df_desc_list[i].loc[("All datasets, weighted"), ] = df_desc_list[i]['Single Trial Correlation'].mul((df_desc_list[i]['Single Trial Correlation']['count'].values.reshape(-1, 1)) / np.sum(df_desc_list[i]['Single Trial Correlation']['count'].values)).sum().values
    # df_desc_list[i].to_csv(f'notebooks/submission_m4/results/validation_pred_description_{i}.csv', index = False)
    df_desc_list[i].to_csv(f'{RUN_FOLDER}/results/validation_pred_description_{i}.csv', index=False)

# get index in folders for LPE10885/2023_10_20 if it exists
true_idx = None
for idx, folder in enumerate(folders):
    if 'LPE10885/2023_10_20' in folder:
        true_idx = idx
        break

num_neurons = df_desc['Single Trial Correlation']['count'].iloc[true_idx].astype(int)

mus = np.zeros((5, num_neurons, 2))
sigmas = np.zeros((5, num_neurons, 2, 2))
jitters = np.zeros((5, num_neurons, 2))
locs = np.zeros((5, num_neurons, 2))

# model.readout._modules

for dataset_name_full in df['dataset_name_full'].unique():
    df_trunc = df.loc[df['dataset_name_full'] == dataset_name_full]
    num_neurons = df_trunc['cell_id'].nunique()
    
    mus = np.zeros((num_models, num_neurons, 2))
    sigmas = np.zeros((num_models, num_neurons, 2, 2))
    jitters = np.zeros((num_models, num_neurons, 2))
    locs = np.zeros((num_models, num_neurons, 2))
    
    for i, model in enumerate(model_list):
        mus[i] = model.readout._modules[dataset_name_full].mu.detach().cpu().numpy().reshape(-1, 2)
        sigmas[i] = model.readout._modules[dataset_name_full].sigma.detach().cpu().numpy().reshape(-1, 2, 2)
        jitters[i] = model.readout._modules[dataset_name_full].jitter.detach().cpu().numpy().reshape(-1, 2)
        locs[i] = mus[i] + jitters[i]

    neuron_stats = {
        'mean': mus.mean(axis=0),
        'cov': sigmas.mean(axis=0),
        'jitter': jitters.mean(axis=0),
        'loc': locs.mean(axis=0),
        'mean_std': mus.std(axis=0),
        'cov_std': sigmas.std(axis=0),
        'jitter_std': jitters.std(axis=0),
        'loc_std': locs.std(axis=0),
        'single_trial_correlation': df_trunc['Single Trial Correlation'].values,
        'cell_id': df_trunc['cell_id'].values
    }
    
    for i in range(num_models):
        neuron_stats[f'mean_{i}'] = mus[i]
        neuron_stats[f'cov_{i}'] = sigmas[i]
        neuron_stats[f'jitter_{i}'] = jitters[i]
        neuron_stats[f'loc_{i}'] = locs[i]

    np.save(f'{RUN_FOLDER}/results/neuron_stats_{dataset_name_full}.npy', neuron_stats)

# if true_idx:
#     df_trunc = df.loc[df['dataset_name_full'] == 'LPE10885-LPE10885_2023_10_20-0']
#     for i, model in enumerate(model_list):
#         mus[i] = model.readout._modules['LPE10885-LPE10885_2023_10_20-0'].mu.detach().cpu().numpy().reshape(-1, 2)
#         sigmas[i] = model.readout._modules['LPE10885-LPE10885_2023_10_20-0'].sigma.detach().cpu().numpy().reshape(-1, 2, 2)
#         jitters[i] = model.readout._modules['LPE10885-LPE10885_2023_10_20-0'].jitter.detach().cpu().numpy().reshape(-1, 2)
#         locs[i] = mus[i] + jitters[i]

#     # raise NotImplementedError("Save as np arrays instead of CSV")

#     df_neuron_stats = pd.DataFrame(columns=['dataset', 'neuron', 'mean', 'cov', 'mean_std', 'cov_std'] + [f'mean_{i}' for i in range(num_models)] + [f'cov_{i}' for i in range(num_models)])

#     df_neuron_stats['dataset'] = df_trunc['dataset']
#     df_neuron_stats['neuron'] = np.repeat(np.arange(num_neurons), len(df_trunc['dataset'].unique()))
#     for i in range(num_models):
#         df_neuron_stats[f'mean_{i}'] = list(mus[i].round(2))
#         df_neuron_stats[f'cov_{i}'] = list(sigmas[i].round(2))
#         df_neuron_stats[f'jitter_{i}'] = list(jitters[i].round(2))
#         df_neuron_stats[f'loc_{i}'] = list(locs[i].round(2))

#     df_neuron_stats['mean'] = list(mus.mean(axis=0).round(2))
#     df_neuron_stats['cov'] = list(sigmas.mean(axis=0).round(2))
#     df_neuron_stats['jitter'] = list(jitters.mean(axis=0).round(2))
#     df_neuron_stats['loc'] = list(locs.mean(axis=0).round(2))

#     df_neuron_stats['mean_std'] = list(mus.std(axis=0).round(2))
#     df_neuron_stats['cov_std'] = list(sigmas.std(axis=0).round(2))
#     df_neuron_stats['jitter_std'] = list(jitters.std(axis=0).round(2))
#     df_neuron_stats['loc_std'] = list(locs.std(axis=0).round(2))

#     df_neuron_stats['single_trial_correlation'] = df_trunc['Single Trial Correlation']
#     df_neuron_stats['cell_id'] = df_trunc['cell_id']

#     # df_neuron_stats.to_csv('notebooks/submission_m4/results/neuron_stats.csv', index = False)
#     df_neuron_stats.to_csv(f'{RUN_FOLDER}/results/neuron_stats.csv', index = False)

# else:
#     print("LPE10885/2023_10_20 not found in folders")


# df_neuron_stats

# df

# Get fit parameters from the readout, ie x and y coordinates of spatial mask


