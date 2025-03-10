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
map_to_ground_truth = run_config['data']['map_to_ground_truth']
gt_session_id = run_config['data']['gt_session_id']
gt_date = run_config['data']['gt_date']

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

if not isinstance(sessions_to_keep, str):
    sessions_to_keep = [tuple(x) for x in sessions_to_keep]
    session_list = [tuple(x) for x in session_list]
    session_list = [x for x in session_list if x in sessions_to_keep]
    session_list = [list(x) for x in session_list]
elif sessions_to_keep == 'all':
    pass

session_list = np.array(session_list)

sessions, nSessions = load_sessions(protocol='IM', session_list=session_list, data_folder = INPUT_FOLDER)

for ises in range(nSessions):    # Load proper data and compute average trial responses:
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

sessions = compute_pairwise_anatomical_distance(sessions)
sessions = smooth_rf(sessions,radius=75,rf_type='Fneu')
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 

# raise Exception('stop here')

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

    plt.clf() # Clears any previous figures
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

areas = [area_of_interest]
sig_thr = 0
r2_thr  = np.inf
# rf_type = 'Fsmooth'
rf_type = 'Ftwin'

os.makedirs(f'{RUN_FOLDER}/Plots/rf_analysis', exist_ok=True)
os.makedirs(f'{RUN_FOLDER}/results/', exist_ok=True)

for ises, dataset_name_full in enumerate(np.sort(df['dataset_name_full'].unique())):
    areas = [area_of_interest]
    sig_thr = 0
    r2_thr  = np.inf
    # rf_type = 'Fsmooth'
    rf_type = 'Ftwin'

    df_trunc = df.loc[df['dataset_name_full'] == dataset_name_full]
    num_neurons = df_trunc['cell_id'].nunique()
    
    mus = np.zeros((num_models, num_neurons, 2))
    sigmas = np.zeros((num_models, num_neurons, 2, 2))
    jitters = np.zeros((num_models, num_neurons, 2))
    locs = np.zeros((num_models, num_neurons, 2))
    
    for i, model in enumerate(model_list):
        try:
            mus[i] = model.readout._modules[dataset_name_full].mu.detach().cpu().numpy().reshape(-1, 2)
        except AttributeError:
            mus[i] = np.zeros((num_neurons, 2))
            print(f"WARNING: unable to get mus for model {i} dataset {dataset_name_full}. Setting to 0")
        try:
            sigmas[i] = model.readout._modules[dataset_name_full].sigma.detach().cpu().numpy().reshape(-1, 2, 2)
        except AttributeError:
            sigmas[i] = np.zeros((num_neurons, 2))
            print(f"WARNING: unable to get sigmas for model {i} dataset {dataset_name_full}. Setting to 0")
        try:
            jitters[i] = model.readout._modules[dataset_name_full].jitter.detach().cpu().numpy().reshape(-1, 2)
        except AttributeError:
            jitters[i] = np.zeros((num_neurons, 2))
            print(f"WARNING: unable to get jitters for model {i} dataset {dataset_name_full}. Setting to 0")
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

    # np.save(f'{RUN_FOLDER}/results/neuron_stats_{dataset_name_full}.npy', neuron_stats)
    with open(f'{RUN_FOLDER}/results/neuron_stats_{dataset_name_full}.pkl', 'wb') as f:
        pkl.dump(neuron_stats, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Plot RFs

    loc_columns = ['loc']
    loc_columns.extend(f'loc_{i}' for i in range(num_models))

    mergedata = pd.DataFrame(list(neuron_stats['loc']), columns=['rf_az_Ftwin', 'rf_el_Ftwin',])
    for i in range(num_models):
        temp_df = pd.DataFrame(list(neuron_stats[f'loc_{i}']), columns=[f'rf_az_Ftwin_{i}', f'rf_el_Ftwin_{i}'])
        mergedata = pd.concat([mergedata, temp_df], axis=1)

    mergedata['cell_id'] = neuron_stats['cell_id']
    sessions[ises].celldata = sessions[ises].celldata.merge(mergedata, on='cell_id')
    sessions[ises].celldata['rf_r2_Ftwin'] = 0
    sessions[ises].celldata['rf_az_Ftwin'] = (sessions[ises].celldata['rf_az_Ftwin']+0.5)*135
    sessions[ises].celldata['rf_el_Ftwin'] = (sessions[ises].celldata['rf_el_Ftwin']+0.5)*62 - 53

    for i in range(num_models):
        sessions[ises].celldata[f'rf_az_Ftwin_{i}'] = (sessions[ises].celldata[f'rf_az_Ftwin_{i}'] + 0.5) * 135
        sessions[ises].celldata[f'rf_el_Ftwin_{i}'] = (sessions[ises].celldata[f'rf_el_Ftwin_{i}'] + 0.5) * 62 - 53

    fig = plot_rf_plane(sessions[ises].celldata,r2_thr=r2_thr,rf_type=rf_type, dataset=ises, area_s_of_interest=area_of_interest) 
    fig.savefig(os.path.join(f'{RUN_FOLDER}/Plots/rf_analysis', f'{area_of_interest}_plane_TwinModel_{rf_type}_{sessions[ises].sessiondata["session_id"][0]}.png'), format = 'png')

    for i in range(num_models):
        fig = plot_rf_plane(sessions[ises].celldata,r2_thr=r2_thr,rf_type=f'{rf_type}', suffix=f'_{i}', dataset=ises, area_s_of_interest=area_of_interest) 
        fig.savefig(os.path.join(f'{RUN_FOLDER}/Plots/rf_analysis', f'{area_of_interest}_plane_TwinModel_{rf_type}_{sessions[ises].sessiondata["session_id"][0]}_model_{i}.png'), format = 'png')

    if map_to_ground_truth:
        gt_dataset_name = f'{gt_session_id}-{gt_session_id}_{gt_date}-0'

        # if dataset_name_full == 'LPE10885-LPE10885_2023_10_20-0':
        if dataset_name_full == gt_dataset_name:
            areas       = [area_of_interest]
            spat_dims   = ['az', 'el']
            clrs_areas  = get_clr_areas(areas)
            r2_thr       = np.inf
            rf_type      = 'F'
            rf_type_twin = 'Ftwin'
            fig,axes     = plt.subplots(len(areas),len(spat_dims),figsize=(6,6))

            c = sns.xkcd_rgb['barney'] if area_of_interest == 'PM' else sns.xkcd_rgb['seaweed']

            # Flatten axes for easier indexing when areas is of length 1
            if len(areas) == 0:
                raise ValueError(f'Have to have at least one area, not {len(areas)}')
            elif len(areas) == 1:
                axes = np.expand_dims(axes, axis=0)  # Convert axes to 2D with shape (2, 1)

            for iarea,area in enumerate(areas):
                for ispat_dim,spat_dim in enumerate(spat_dims):
                    idx         = (sessions[ises].celldata['roi_name'] == area) & (sessions[ises].celldata['rf_r2_' + rf_type] < r2_thr)
                    x = sessions[ises].celldata[f'rf_{spat_dim}_{rf_type}'][idx]
                    y = sessions[ises].celldata[f'rf_{spat_dim}_{rf_type_twin}'][idx]

                    # sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=7,c=clrs_areas[iarea],alpha=0.5)
                    sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=7,c=c,alpha=0.5)
                    axes[iarea,ispat_dim].set_title(f'{area} {spat_dim}',fontsize=12)
                    axes[iarea,ispat_dim].set_xlabel('Sparse Noise (deg)',fontsize=9)
                    axes[iarea,ispat_dim].set_ylabel(f'Dig. Twin Model',fontsize=9)
                    # if spat_dim == 'az':
                    #     axes[iarea,ispat_dim].set_xlim([-50,135])
                    #     axes[iarea,ispat_dim].set_ylim([-50,135])
                    #     # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
                    # elif spat_dim == 'el':
                    #     axes[iarea,ispat_dim].set_xlim([-150.2,150.2])
                    #     axes[iarea,ispat_dim].set_ylim([-150.2,150.2])
                        # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
                    idx = (~np.isnan(x)) & (~np.isnan(y))
                    x =  x[idx]
                    y =  y[idx]
                    # print(f'x min: {min(x) if len(x) > 0 else "None"}')
                    # print(f'x max: {max(x) if len(x) > 0 else "None"}')
                    # print(f'y min: {min(y) if len(y) > 0 else "None"}')
                    # print(f'y max: {max(y) if len(y) > 0 else "None"}')
                    if len(x) > 0:
                        axes[iarea,ispat_dim].set_xlim([int(min(x) - 10), int(max(x) + 10)])
                    if len(y) > 0:
                        axes[iarea,ispat_dim].set_ylim([int(min(y) - 10), int(max(y) + 10)])
                    # axes[iarea,ispat_dim].text(x=0,y=0.1,s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
                    if len(x) > 0 and len(y) > 0:
                        axes[iarea,ispat_dim].text(x=int(min(x) - 5),y=int(min(y) - 5),s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
            fig.suptitle(f'Mean of 5 models')
            plt.tight_layout()
            fig.savefig(os.path.join(f'{RUN_FOLDER}/Plots/rf_analysis', f'Alignment_TwinGaussLoc_RF_{rf_type}_{sessions[ises].sessiondata["session_id"][0]}.png'), format='png')

            for i in range(num_models):
                fig,axes     = plt.subplots(len(areas),len(spat_dims),figsize=(6,6))
                
                # Flatten axes for easier indexing when areas is of length 1
                if len(areas) == 0:
                    raise ValueError(f'Have to have at least one area, not {len(areas)}')
                elif len(areas) == 1:
                    axes = np.expand_dims(axes, axis=0)  # Convert axes to 2D with shape (2, 1)
                for iarea,area in enumerate(areas):
                    for ispat_dim,spat_dim in enumerate(spat_dims):
                        idx         = (sessions[ises].celldata['roi_name'] == area) & (sessions[ises].celldata['rf_r2_' + rf_type] < r2_thr)
                        x = sessions[ises].celldata[f'rf_{spat_dim}_{rf_type}'][idx]
                        y = sessions[ises].celldata[f'rf_{spat_dim}_{rf_type_twin}_{i}'][idx]

                        # sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=7,c=clrs_areas[iarea],alpha=0.5)
                        sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=7,c=c,alpha=0.5)
                        axes[iarea,ispat_dim].set_title(f'{area} {spat_dim} Model {i}',fontsize=12)
                        axes[iarea,ispat_dim].set_xlabel('Sparse Noise (deg)',fontsize=9)
                        axes[iarea,ispat_dim].set_ylabel(f'Dig. Twin Model {i}',fontsize=9)
                        # if spat_dim == 'az':
                        #     axes[iarea,ispat_dim].set_xlim([-50,135])
                        #     axes[iarea,ispat_dim].set_ylim([-50,135])
                        #     # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
                        # elif spat_dim == 'el':
                        #     axes[iarea,ispat_dim].set_xlim([-150.2,150.2])
                        #     axes[iarea,ispat_dim].set_ylim([-150.2,150.2])
                        #     # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
                        idx = (~np.isnan(x)) & (~np.isnan(y))
                        x =  x[idx]
                        y =  y[idx]
                        # print(f'x min: {min(x) if len(x) > 0 else "None"}')
                        # print(f'x max: {max(x) if len(x) > 0 else "None"}')
                        # print(f'y min: {min(y) if len(y) > 0 else "None"}')
                        # print(f'y max: {max(y) if len(y) > 0 else "None"}')
                        if len(x) > 0:
                            axes[iarea,ispat_dim].set_xlim([int(min(x) - 10), int(max(x) + 10)])
                        if len(y) > 0:
                            axes[iarea,ispat_dim].set_ylim([int(min(y) - 10), int(max(y) + 10)])
                        # axes[iarea,ispat_dim].text(x=0,y=0.1,s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
                        if len(x) > 0 and len(y) > 0:
                            axes[iarea,ispat_dim].text(x=int(min(x) - 5),y=int(min(y) - 5),s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
                plt.suptitle(f'Model {i}')
                plt.tight_layout()
                fig.savefig(os.path.join(f'{RUN_FOLDER}/Plots/rf_analysis', f'Alignment_TwinGaussLoc_RF_{rf_type}_{sessions[ises].sessiondata["session_id"][0]}_model_{i}.png'), format='png')