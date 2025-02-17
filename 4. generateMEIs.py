import sys
import os
import torch
import numpy as np
import pandas as pd
import mei.legacy
import matplotlib.pyplot as plt
import seaborn as sns
import sensorium
import warnings
from tqdm.auto import tqdm
from nnfabrik.builder import get_data, get_model
from gradient_ascent import gradient_ascent
from sensorium.utility import get_signal_correlations
from sensorium.utility.measure_helpers import get_df_for_scores
from sensorium.utility.training import read_config
from sensorium.models.ensemble import EnsemblePrediction
from neuralpredictors.measures.np_functions import corr
from sensorium.utility.submission import get_data_filetree_loader
from PIL import Image
from scipy import stats

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
data_key = f"{run_config['MEIs']['session_id']}-{run_config['MEIs']['session_id']}_{run_config['MEIs']['session_date']}-0"
INPUT_FOLDER = run_config['data']['INPUT_FOLDER']
data_basepath = f'{INPUT_FOLDER}/'
area_of_interest = run_config['current_vals']['data']['area_of_interest']
tier = run_config['MEIs']['tier']
mei_shape = run_config['MEIs']['shape']
num_models = run_config['dev']['num_models']
num_meis = run_config['MEIs']['num_meis']
num_labeled_cells = run_config['MEIs']['num_labeled_cells']
session_id = run_config['MEIs']['session_id']
session_date = run_config['MEIs']['session_date']

print(f'Starting MEI generation for {RUN_NAME}')

# ## Restart Kernel after mei-module installation!

warnings.filterwarnings('ignore')

seed=31415
# data_key_aut = "29027-6-17-1-6-5"
# data_key_wt = "29028-1-17-1-6-5"
# data_key_sens2 = "23964-4-22"
# autistic_mouse_dataPath = "../data/new_data2023/static29027-6-17-1-6-5-GrayImageNetFrame2-7bed7f7379d99271be5d144e5e59a8e7.zip"
# wildtype_mouse_dataPath = "../data/new_data2023/static29028-1-17-1-6-5-GrayImageNetFrame2-7bed7f7379d99271be5d144e5e59a8e7.zip"
# sens2_dataPath = "../data/sensorium_data2022/static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"

# Loading config only for ensemble 0, because all 5 models have the same config (except
# for the seed and dataloader train/validation split)

# config_file = 'config_m4_ens0.yaml'
config_file = f'{RUN_FOLDER}/config_m4_ens0/config.yaml'
config = read_config(config_file)
config['model_config']['data_path'] = f'{RUN_FOLDER}/data'
print(config)

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

# dataset_fn = 'sensorium.datasets.static_loaders'


dataset_fn = config['dataset_fn']  # 'sensorium.datasets.static_loaders'
dataset_config = {'paths': folders,  # filenames,
                  **config['dataset_config'],
                  }

dataloaders = get_data(dataset_fn, dataset_config)

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

ensemble = EnsemblePrediction(model_list, mode='mean')

print("Getting signal correlations")
correlation_to_average = get_signal_correlations(ensemble, dataloaders, tier='test', device='cuda', as_dict=True)

df_cta = get_df_for_scores(session_dict=correlation_to_average, measure_attribute="Correlation to Average")

config_mei = dict(
    initial={"path": "mei.initial.RandomNormal"},
    optimizer={"path": "torch.optim.SGD", "kwargs": {"lr": 1}},
    # transform={"path": "C:\\Users\\asimo\\Documents\\BCCN\\Lab Rotations\\Petreanu Lab\\adrian_sensorium\\notebooks\\submission_m4\\transform.only_keep_1st_dimension"},
    # transform={"path": "transform.only_keep_1st_dimension", "kwargs": {"mei": torch.zeros(1, 4, 64, 64), "i_iteration": 0}},
    # transform={"path": "transform.only_keep_1st_dimension"},#, "kwargs": {"mei": torch.zeros(1, 4, 64, 64), "i_iteration": 0}},
    transform={"path": "transform.OnlyKeep1stDimension"},# "kwargs": {"mei": None, "i_iteration": None}},
    precondition={"path": "mei.legacy.ops.GaussianBlur", "kwargs": {"sigma": 1}},
    postprocessing={"path": "mei.legacy.ops.ChangeNorm", "kwargs": {"norm": 7.5}},
    transparency_weight=0.0,
    stopper={"path": "mei.stoppers.NumIterations", "kwargs": {"num_iterations": 1000}},
    objectives=[
        {"path": "mei.objectives.EvaluationObjective", "kwargs": {"interval": 10}}
    ],
    device="cuda"
)

df_cta = df_cta.loc[df_cta['dataset'] == data_key].reset_index(drop=True)

top200units = df_cta.sort_values(['Correlation to Average'], ascending=False).reset_index()[:200]['index'].to_list()
top40units = df_cta.sort_values(['Correlation to Average'], ascending=False).reset_index()[:40]['index'].to_list()

ensemble = ensemble.eval()

pupil_center_config = {"pupil_center": torch.from_numpy(stats.mode([np.mean(np.array(list(dataloaders[i][data_key].dataset._cache['pupil_center'].values())), axis=0) for i in dataloaders]).mode).to(torch.float32).to("cuda:0")}

data_path = os.path.join(data_basepath, data_key.split('-')[1].split('_')[0] + '/' + '_'.join(data_key.split('-')[1].split('_')[1:]))
celldata = pd.read_csv(data_path + '/celldata.csv')
celldata = celldata.loc[celldata['roi_name'] == area_of_interest] if area_of_interest is not None else celldata
assert len(df_cta[df_cta['dataset'] == data_key]) == len(celldata), f"Length of df_cta and celldata not equal, {len(df_cta[df_cta['dataset'] == data_key])} != {len(celldata)} of {data_key}"
df_cta.loc[df_cta['dataset'] == data_key, 'labeled'] = celldata['redcell'].astype(bool).values
df_cta.loc[df_cta['dataset'] == data_key, 'cell_id'] = celldata['cell_id'].values

for loader_key, dataloader in dataloaders[tier].items():
    if loader_key != data_key:
        continue
    trial_indices, image_ids, neuron_ids, responses = get_data_filetree_loader(
        dataloader=dataloader, tier=tier
    )

oracles, data = [], []
# for inputs, *_, outputs in dataloaders[tier][data_key]:
for i in dataloaders[tier][data_key]:
    outputs = i.responses.cpu().numpy()
    r, n = outputs.shape  # responses X neurons
    mu = outputs.mean(axis=0, keepdims=True)
    oracle = (mu - outputs / r) * r / (r - 1)
    oracles.append(oracle)
    data.append(outputs)
if len(data) == 0:
    raise ValueError('Found no oracle trials!')

# Pearson correlation
pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

# Spearman correlation
data_rank = np.empty(np.vstack(data).shape)
oracles_rank = np.empty(np.vstack(oracles).shape)
for i in range(np.vstack(data).shape[1]):
    data_rank[:, i] = np.argsort(np.argsort(np.vstack(data)[:, i]))
    oracles_rank[:, i] = np.argsort(np.argsort(np.vstack(oracles)[:, i]))
spearman = corr(data_rank, oracles_rank, axis=0)

# oracle_scores_pearson = np.mean(pearson, axis=0)
# oracle_scores_spearman = np.mean(spearman, axis=0)
oracle_scores_pearson = pearson
oracle_scores_spearman = spearman
oracle_scores = oracle_scores_pearson  # or use oracle_scores_spearman if preferred, but Finz et al. uses Pearson

# 1. Select cells within top 50% of oracle scores

selected_neurons = np.argsort(oracle_scores)[::-1][:len(oracle_scores)//2]

# 2. Exclude neurons within 10um of the scanning fields

# Min and max values of the scanning fields
x_min = 0
x_max = 600
y_min = 0
y_max = 600

# Calculate the max values allowed for the neurons
x_min = x_min + 10
x_max = x_max - 10
y_min = y_min + 10
y_max = y_max - 10

selected_neurons = selected_neurons[np.where((celldata['xloc'].values[selected_neurons] > x_min) & (celldata['xloc'].values[selected_neurons] < x_max) & (celldata['yloc'].values[selected_neurons] > y_min) & (celldata['yloc'].values[selected_neurons] < y_max))[0]]

# 3. Select cells within top 30% of oracle scores of previous selection. Is this step necessary?

# selected_neurons = selected_neurons[np.argsort(oracle_scores[selected_neurons])[::-1][:len(selected_neurons)//3]]

# 4. Iterate for neurons in order of decreasing oracle scores, excluding neurons that are within 20um of it

neurons_to_exclude = []
final_neurons = []
for i in selected_neurons:
    if i in neurons_to_exclude:
        continue
    final_neurons.append(i)
    neurons_to_exclude.extend(np.where(np.linalg.norm(celldata[['xloc', 'yloc', 'depth']].values - celldata[['xloc', 'yloc', 'depth']].values[i], axis=1) < 20)[0])

# 5. Select max num_meis best neurons of which at least 10 are labeled
celldata['index_reset'] = np.arange(len(celldata))
final_neurons_labeled = celldata.iloc[final_neurons].loc[(celldata['redcell'] == True)]['index_reset'].values

if len(final_neurons_labeled) < 10:
    print(f'WARNING: Less than 10 labeled neurons available, {len(final_neurons_labeled)} available. Continuing on...')

final_selection = []
labeled_count = 0

for i in final_neurons:
    if len(final_selection) >= num_meis:
        break
    if i in final_neurons_labeled:
        labeled_count += 1
    final_selection.append(i)

if labeled_count < num_labeled_cells:
    additional_needed = num_labeled_cells - labeled_count
    remaining_labeled = [i for i in final_neurons_labeled if i not in final_selection]

    final_selection.extend(remaining_labeled[:additional_needed])

# 6. Assert that at least num_labeled_cells labeled neurons are selected

# assert celldata.loc[final_neurons, 'redcell'].sum() >= 10, f"Less than 10 labeled neurons selected, {celldata.loc[final_neurons, 'redcell'].sum()} selected"
if celldata.iloc[final_selection]['redcell'].sum() < num_labeled_cells:
    print(f"WARNING: Less than {num_labeled_cells} labeled neurons selected, {celldata.iloc[final_neurons]['redcell'].sum()} selected for MEI generation")
else:
    print(f"Selected {celldata.iloc[final_selection]['redcell'].sum()} labeled neurons for MEI generation")

cell_ids = df_cta.iloc[final_selection]['cell_id'].values

# save cell_ids, final_neurons
df_cell_ids = pd.DataFrame({'cell_id': cell_ids, 'neuron_idx': final_selection})
df_cell_ids.to_csv(f'{RUN_FOLDER}/results/cell_ids.csv', index=False)

meis = []

mei_shape_start = (1, 4) # We prepend this because there's 4 input channels: 1 image and 3 behavioral
mei_shape_start = list(mei_shape_start)
mei_generation_shape = mei_shape_start.copy()
mei_generation_shape.extend(list(mei_shape))

print(f'Generating MEIs with the following shape: {mei_generation_shape}')
print(f'Generating {len(final_selection)} best MEIs out of {len(final_neurons)} neurons selected')
for i in tqdm(final_selection):
    mei_out, _, _ = gradient_ascent(ensemble, config_mei, data_key=data_key, unit=i, seed=seed, shape=tuple(mei_generation_shape), model_config=pupil_center_config) # need to pass all dimensions, but all except the first 1 are set to 0 in the transform
    meis.append(mei_out)
# torch.save(meis, "MEIs/meis.pth")
torch.save(meis, f'{RUN_FOLDER}/meis_top{len(final_selection)}_ensemble.pth')

# Save MEIs in Bonsai format
print('Saving MEIs in Bonsai format')
os.makedirs(f'{RUN_FOLDER}/MEI_Bonsai_images', exist_ok=True)

for imei, mei_out in enumerate(meis):
    mei_out = np.array(mei_out[0, 0, ...])
    mei_out = (mei_out + 1) / 2
    mei_out = np.concatenate((np.full(mei_shape, 0.5),mei_out), axis=1) #add left part of the screen
    mei_out = (mei_out * 255).astype(np.uint8)
    # np.save(os.path.join(outdir,'%d.jpg' % imei),mei_out)
    img = Image.fromarray(mei_out)
    img.save(os.path.join(f'{RUN_FOLDER}/MEI_Bonsai_images','%s.jpg' % cell_ids[imei]), format='JPEG')

    if run_config['MEIs']['also_output_to_local']:
        img.save(os.path.join(run_config['MEIs']['local_output_folder'],'%s.jpg' % cell_ids[imei]), format='JPEG')

fig, axes = plt.subplots(8,5, figsize=(20,20), dpi=300)
fig.suptitle("Mouse MEIs", y=0.91, fontsize=50)
for i in tqdm(range(8)):
    for j in range(5):
        index = i * 5 + j
        # axes[i, j].imshow(meis[index].reshape(4, 68, 135).mean(0), cmap="gray")#, vmin=-1, vmax=1)
        axes[i, j].imshow(meis[index][0, 0, ...], cmap="gray")#, vmin=-1, vmax=1)
        axes[i, j].spines['top'].set_color('black')
        axes[i, j].spines['bottom'].set_color('black')
        axes[i, j].spines['left'].set_color('black')
        axes[i, j].spines['right'].set_color('black')
        axes[i, j].spines['top'].set_linewidth(1)
        axes[i, j].spines['bottom'].set_linewidth(1)
        axes[i, j].spines['left'].set_linewidth(1)
        axes[i, j].spines['right'].set_linewidth(1)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.subplots_adjust(wspace=-0.25, hspace=-0.1)
# os.makedirs("Plots", exist_ok=True)
# plt.savefig("Plots/MouseMEIsTop200.png", dpi=300)
os.makedirs(f'{RUN_FOLDER}/Plots', exist_ok=True)
plt.savefig(f'{RUN_FOLDER}/Plots/MouseMEIsTop40.png', dpi=300)
# plt.show()

for i, model in enumerate(model_list):
    model = model.eval()
    model_list[i] = model

for model_idx, model in enumerate(model_list):
    print(f"Generating MEIs for model {model_idx}")
    meis = []
    for i in tqdm(final_selection[:15]):
        mei_out, _, _ = gradient_ascent(model, config_mei, data_key=data_key, unit=i, seed=seed, shape=tuple(mei_generation_shape)) # need to pass all dimensions, but all except the first 1 are set to 0 in the transform
        meis.append(mei_out)
    # torch.save(meis, f"MEIs/meis_model_{model_idx}.pth")
    torch.save(meis, f'{RUN_FOLDER}/meis_model_{model_idx}.pth')

# for k in range(4):
#     fig, axes = plt.subplots(20,10, figsize=(20,20), dpi=300)
#     fig.suptitle(f"Mouse MEIs Channel {k}", y=0.91, fontsize=50)
#     for i in tqdm(range(20)):
#         for j in range(10):
#             index = i * 10 + j
#             axes[i, j].imshow(meis[index].reshape(4, 64, 96)[k, :, :], cmap="gray")#, vmin=-1, vmax=1)
#             axes[i, j].spines['top'].set_color('black')
#             axes[i, j].spines['bottom'].set_color('black')
#             axes[i, j].spines['left'].set_color('black')
#             axes[i, j].spines['right'].set_color('black')
#             axes[i, j].spines['top'].set_linewidth(1)
#             axes[i, j].spines['bottom'].set_linewidth(1)
#             axes[i, j].spines['left'].set_linewidth(1)
#             axes[i, j].spines['right'].set_linewidth(1)
#             axes[i, j].set_xticks([])
#             axes[i, j].set_yticks([])
#     plt.subplots_adjust(wspace=-0.25, hspace=-0.1)
#     os.makedirs("Plots", exist_ok=True)
#     plt.savefig(f"Plots/MouseMEIsTop200Channel{k}.png", dpi=300)
#     plt.show()

for k in range(num_models):
    # meis = torch.load(f"MEIs/meis_model_{k}.pth")
    meis = torch.load(f'{RUN_FOLDER}/meis_model_{k}.pth')
    fig, axes = plt.subplots(3,5, figsize=(20,20), dpi=300)
    fig.suptitle(f"Mouse MEIs model {k}", y=0.91, fontsize=50)
    for i in tqdm(range(3)):
        for j in range(5):
            index = i * 5 + j
            axes[i, j].imshow(meis[index].reshape(mei_generation_shape[1:])[0, :, :], cmap="gray")#, vmin=-1, vmax=1)
            axes[i, j].spines['top'].set_color('black')
            axes[i, j].spines['bottom'].set_color('black')
            axes[i, j].spines['left'].set_color('black')
            axes[i, j].spines['right'].set_color('black')
            axes[i, j].spines['top'].set_linewidth(1)
            axes[i, j].spines['bottom'].set_linewidth(1)
            axes[i, j].spines['left'].set_linewidth(1)
            axes[i, j].spines['right'].set_linewidth(1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.subplots_adjust(wspace=-0.25, hspace=-0.1)
    # os.makedirs("Plots", exist_ok=True)
    # plt.savefig(f"Plots/MouseMEIsTop200Model{k}.png", dpi=300)
    os.makedirs(f'{RUN_FOLDER}/Plots', exist_ok=True)
    plt.savefig(f'{RUN_FOLDER}/Plots/MouseMEIsTop40Model{k}.png', dpi=300)
    # plt.show()

meis_list = []

for i in range(num_models):
    # meis_list.append(torch.load(f"MEIs/meis_model_{i}.pth"))
    meis_list.append(torch.load(f'{RUN_FOLDER}/meis_model_{i}.pth'))

meis_list = [torch.stack(meis, dim=0) for meis in meis_list]
meis_list = torch.stack(meis_list, dim=0)

avg_meis = meis_list.mean(dim=0)

fig, axes = plt.subplots(3,5, figsize=(20,20), dpi=300)
fig.suptitle("Mouse MEIs Average", y=0.91, fontsize=50)
for i in tqdm(range(3)):
    for j in range(num_models):
        index = i * 5 + j
        axes[i, j].imshow(avg_meis[index][0, 0, :, :], cmap="gray")#, vmin=-1, vmax=1)
        axes[i, j].spines['top'].set_color('black')
        axes[i, j].spines['bottom'].set_color('black')
        axes[i, j].spines['left'].set_color('black')
        axes[i, j].spines['right'].set_color('black')
        axes[i, j].spines['top'].set_linewidth(1)
        axes[i, j].spines['bottom'].set_linewidth(1)
        axes[i, j].spines['left'].set_linewidth(1)
        axes[i, j].spines['right'].set_linewidth(1)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.subplots_adjust(wspace=-0.25, hspace=-0.1)
# os.makedirs("Plots", exist_ok=True)
# plt.savefig("Plots/MouseMEIsTop200Average.png", dpi=300)
os.makedirs(f'{RUN_FOLDER}/Plots', exist_ok=True)
plt.savefig(f'{RUN_FOLDER}/Plots/MouseMEIsTop40Average.png', dpi=300)
# plt.show()
