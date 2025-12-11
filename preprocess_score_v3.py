# Note: this version implements frame correspondance, snr, l1 loss, etc., and outputs only scores.
import os
import logging
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src import data
from torch import Tensor
from tqdm import tqdm
import torch
from src.model.utils.tools import pack_to_render
logger = logging.getLogger(__name__)
import numpy as np
import smplx
import matplotlib.pyplot as plt
import src.launch.prepare  # registers custom OmegaConf/Hydra resolvers

def plot_list_distribution(data, title):
    """
    Plot a histogram for a list of values.

    Args:
        data (list): List of numeric values to plot.
    """
    # Clip values to a maximum of 10 for visualization
    data = [min(x, 10) for x in data]
    # Clear current figure and draw histogram
    plt.clf()
    plt.hist(data, bins=20, edgecolor='black')
    # Axis labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Continuous Value Distribution')
    # Save to file
    plt.savefig(f"{title}.png")
# calculate the distance using the metric proposed in the paper
# 

@hydra.main(version_base=None, config_path="configs", config_name="motionfix_eval")
def _render_vids(cfg: DictConfig) -> None:
    return render_vids(cfg)

import torch

class RunningStats:
    def __init__(self):
        self.n = 0  # Number of elements
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from the current mean

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self):
        if self.n < 2:
            return float('nan')  # Standard deviation is not defined for n < 2
        return (self.M2 / (self.n - 1)) ** 0.5

def calculate_snr(distance_list):
    # the larger the better
    # could set threshold to 1
    if len(distance_list) < 5:
        raise ValueError("List must contain at least 5 elements")

    sorted_distances = sorted(distance_list, reverse=True)
    
    longest_avg = sum(sorted_distances[-10:]) / 10
    shortest_avg = sum(sorted_distances[:10]) / 10
    
    snr = longest_avg / shortest_avg
    return snr

def moving_average(sequence, window_size = 10):
    return np.convolve(sequence, np.ones(window_size) / window_size, mode='same')

def motion_retrieval(motion_source, motion_target):
    # source: T, 135 [rot, trans]
    # target: T_target, 135 [rot, trans]
    T_s = motion_source.shape[0]
    T_t = motion_target.shape[0]
    # Collect MSE loss for each window in motion_source
    losses = []
    window_size = 15

    # Iterate through all sliding windows in motion_source
    for i in range(T_s):
        # Extract the current window from motion_source
        source_window = motion_source[i:i + 1]

        # Find the best matching window in motion_target
        best_mse = float('inf')  # Initialize with a large value

        # Iterate through all sliding windows in motion_target
        target_window_center = int(i*T_t/T_s)
        for j in range(max(target_window_center - window_size, 0), min(target_window_center + window_size, T_t)):
            target_window = motion_target[j:j + 1]
            
            # Calculate MSE loss or L1 loss
            # mse_loss = np.mean((source_window - target_window) ** 2)
            mse_loss = np.mean(np.abs(source_window - target_window))
            
            # Update the best MSE if a lower loss is found
            if mse_loss < best_mse:
                best_mse = mse_loss

        # Append the best MSE for the current source window
        losses.append(-best_mse)

    return moving_average(losses)

def motion_retrieval_balanced(smpl_layer, dic_source, dic_target): 
    # Balance the loss between joint space and raw feature space
    device = dic_target.device
    target_motion = pack_to_render(dic_target[:,3:], dic_target[:,:3])
    source_motion = pack_to_render(dic_source[:,3:], dic_source[:,:3])
    T_s = source_motion['body_pose'].shape[0]
    T_t = target_motion['body_pose'].shape[0]
    with torch.no_grad():
        # Zeros mean no rotation for the last 6 joint slots
        source_out = smpl_layer(body_pose = torch.cat([source_motion['body_pose'], torch.zeros((T_s, 6)).to(device)], dim=1), global_orient=source_motion['body_orient']) 
        target_out = smpl_layer(body_pose = torch.cat([target_motion['body_pose'], torch.zeros((T_t, 6)).to(device)], dim=1), global_orient=target_motion['body_orient']) 
    source_joints = source_out.joints
    target_joints = target_out.joints # shape: (B, J, 3)

    # print(source_joints.shape)
    # print(target_joints.shape)
    source_joints_flatten = source_joints.view(T_s, -1)
    target_joints_flatten = target_joints.view(T_t, -1)

    # NOTE: instead of adaptive, we manually balance the loss. Adaptive loss doesn't make sense
    losses_1 = motion_retrieval(source_joints_flatten.cpu().numpy(), target_joints_flatten.cpu().numpy())
    losses_2 = motion_retrieval(dic_source[:, 9:].cpu().numpy(), dic_target[:, 9:].cpu().numpy())
    weight_1 = 1
    weight_2 = 1
    assert len(losses_1) == len(losses_2)

    losses = [weight_1*losses_1[i] + weight_2* losses_2[i] for i in range(len(losses_1))]
    # return: motion torch losses, mean for reference
    snr = calculate_snr(losses)
    # return torch.tensor(losses), np.mean(losses_1), np.mean(losses_2)
    return torch.tensor(losses), np.mean(losses_1), np.mean(losses_2), snr

def min_max_normalization_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # Apply min-max normalization formula
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-6)
    
    return normalized_tensor

def render_vids(newcfg: DictConfig) -> None:
    from pathlib import Path
    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")    
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    # Minimal: load only model for data2motion utility; no generation/sampling.
    from hydra.utils import instantiate
    import numpy as np
    logger.info("Loading model (for data2motion only)")
    from src.model.base_diffusion import MD
    # Load the last checkpoint
    model = MD.load_from_checkpoint(last_ckpt_path, renderer=None, strict=False)
    model.eval()
    model.freeze()
    logger.info(f"Model '{cfg.model.modelname}' loaded (no generation)")


    data_module = instantiate(cfg.data, amt_only=True,
                              load_splits=['test', 'val','train'])

    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True
    # load the test set and collate it properly
    features_to_load = data_module.dataset['test'].load_feats
    test_dataset = data_module.dataset['test'] + data_module.dataset['val'] + data_module.dataset['train']
    # 6223 -> 6730
    print(len(test_dataset))
    print(len(data_module.dataset['test']))
    print(len(data_module.dataset['train']))
    from src.data.tools.collate import collate_batch_last_padding
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)

    subset = []
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=8,
                                             batch_size=128,
                                             collate_fn=collate_fn, drop_last=False)
    ds_iterator = testloader 

    logger.info(f'Evaluation Set length:{len(test_dataset)}')
    # P = PATH/TO/FOLDER/THAT/HAS/SMPL_NEUTRAL.pkl
    smpl_layer = smplx.SMPL(model_path=r"P").to(model.device)

    data_score_dict = {}
    use_rate = RunningStats()
    snr_list = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(ds_iterator)):
            target_lens = batch['length_target']
            source_lens = batch['length_source']
            keyids = batch['id']
            no_of_motions = len(keyids)

            list_of_sources = []
            list_of_targets = []
            feature_types = ['body_transl_delta_pelv', 'body_orient_xy', 'z_orient_delta', 'body_pose', 'body_joints_local_wo_z_rot']
            for i in range(no_of_motions):
                single_batch = {key: batch[key][i:i+1] for key in batch.keys() if 'source' in key}
                single_batch = { k: v.to(model.device) if torch.is_tensor(v) else v for k, v in single_batch.items() }
                source_single_batch = model.data2motion(single_batch, feature_types, mot='source')
                list_of_sources.append(source_single_batch)

            for i in range(no_of_motions):
                single_batch = {key: batch[key][i:i+1] for key in batch.keys() if 'target' in key}
                single_batch = { k: v.to(model.device) if torch.is_tensor(v) else v for k, v in single_batch.items() }
                target_single_batch = model.data2motion(single_batch, feature_types, mot='target')
                list_of_targets.append(target_single_batch)

            source_batch = torch.cat(list_of_sources, dim=0)
            target_batch = torch.cat(list_of_targets, dim=0)

            for i in range(len(keyids)):
                T_s = source_lens[i]
                T_t = target_lens[i]
                motion_source = source_batch[i, :source_lens[i]]
                motion_target = target_batch[i, :target_lens[i]]
                data_id = keyids[i]

                data_pre, _, _, snr = motion_retrieval_balanced(smpl_layer, motion_source, motion_target)
                snr_list.append(snr)
                data_score_dict[data_id] = {}
                if snr < 2 or T_s < 20 or T_t < 20:
                    data_score_dict[data_id]['use_loss'] = False
                    use_rate.update(0)
                else:
                    data_score_dict[data_id]['use_loss'] = True
                    use_rate.update(1)
                data_score_dict[data_id]['score_original'] = data_pre
                norm_pre = min_max_normalization_tensor(data_pre)
                data_score_dict[data_id]['score'] = norm_pre

    plot_list_distribution(snr_list, 'snr')
    np.savez('scores_v3.npz', data_score_dict)
    print(use_rate.get_mean())
    logger.info("Scores saved to scores_v3.npz")

if __name__ == '__main__':
    _render_vids()
