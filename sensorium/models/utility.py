import copy
import numpy as np
import os
import torch


def prepare_grid(grid_mean_predictor, dataloaders, **kwargs):
    """
    Utility function for using the neurons cortical coordinates
    to guide the readout locations in image space.

    Args:
        grid_mean_predictor (dict): config dictionary, for example:
          {'type': 'cortex',
           'input_dimensions': 2,
           'hidden_layers': 1,
           'hidden_features': 30,
           'final_tanh': True}

        dataloaders: a dictionary of dataloaders, one PyTorch DataLoader per session
            in the format {'data_key': dataloader object, .. }
    Returns:
        grid_mean_predictor (dict): config dictionary
        grid_mean_predictor_type (str): type of the information that is being used for
            the grid positition estimator
        source_grids (dict): a grid of points for each data_key

    """
    if grid_mean_predictor is None:
        grid_mean_predictor_type = None
        source_grids = None
    else:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")

        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {
                k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                for k, v in dataloaders.items()
            }
        elif grid_mean_predictor_type == "RF":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            input_path = kwargs.get("input_path", None)
            if input_path is not None:
                source_grids = {
                    k: torch.load(os.path.join(input_path, 
                                    k.split('-')[0], # eg, LPE10885
                                    '_'.join(k.split('-')[1].split('_')[1:]), # eg, 2020_03_20
                                    'meta', 'neurons',
                                    'rf_data.pt'
                                    ) ).cpu().detach().numpy()[:, :input_dim]
                                    for k, v in dataloaders.items()
                }
            else:
                raise ValueError("input_path must be provided for RF grid_mean_predictor")
        else:
            raise ValueError(f"Unknown grid_mean_predictor_type: {grid_mean_predictor_type}")
    return grid_mean_predictor, grid_mean_predictor_type, source_grids