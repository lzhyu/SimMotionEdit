
import hydra
from pathlib import Path
from tmr_evaluator.motion2motion_retr import read_config
import logging
logger = logging.getLogger(__name__)

import pytorch_lightning as pl
import numpy as np
from hydra.utils import instantiate
from src.tmr.load_model import load_model_from_cfg
from src.tmr.metrics import all_contrastive_metrics_mot2mot, print_latex_metrics_m2m
from omegaconf import DictConfig
from omegaconf import OmegaConf
import src.launch.prepare 
from src.data.tools.collate import collate_batch_last_padding
import torch
from tmr_evaluator.motion2motion_retr import length_to_mask
# Step 1: load TMR model

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def masked_loss(loss, mask):
    # loss: B, T
    # mask: B, T
    masked_loss = loss * mask  # 只保留 mask 位置的损失

    # 计算平均 masked loss
    # 1. 计算有效元素数量，避免除以零
    num_valid_elements = mask.sum()  # 有效元素的总数
    masked_loss_mean = masked_loss.sum() / (num_valid_elements + 1)
    return masked_loss_mean

def load_tmr_model():
    protocol = ['normal', 'batches']
    device = 'cpu'
    run_dir = 'eval-deps'
    ckpt_name = 'last'
    batch_size = 256
 
    protocols = protocol
    dataset = 'motionfix' # motionfix
    sets = 'test' # val all
    # save_dir = os.path.join(run_dir, "motionfix/contrastive_metrics")
    # os.makedirs(save_dir, exist_ok=True)

    # Load last config
    curdir = Path("/depot/bera89/data/li5280/project/motionfix")
    cfg = read_config(curdir / run_dir)
    logger.info("Loading the evaluation TMR model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    return model



# STEP 2: load dataset

def load_testloader(cfg: DictConfig):
    # What do you want?
    exp_folder = Path("/depot/bera89/data/li5280/project/motionfix/experiments/tmed")
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, cfg)
    data_module = instantiate(cfg.data, amt_only=True,
                            load_splits=['test', 'val'])
    # load the test set and collate it properly
    features_to_load = data_module.dataset['test'].load_feats
    SMPL_feats = ['body_transl', 'body_pose', 'body_orient']
    for feat in SMPL_feats:
        if feat not in features_to_load:
            data_module.dataset['test'].load_feats.append(feat)
    print(features_to_load)

    # TODO: change features of including SMPL features
    test_dataset = data_module.dataset['test'] + data_module.dataset['val']
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=8,
                                             batch_size=128,
                                             collate_fn=collate_fn)
    return testloader

from src.data.features import _get_body_transl_delta_pelv_infer
def batch_to_smpl(batch, normalizer, mot_from='source'):
    # batch: dict
    # return: padded batch with lengths
    # simple concatenation of the body pose and body transl
    # trans_delta, body_pose_6d, global_orient_6d
    smpl_keys = ['body_transl', 'body_orient', 'body_pose']
    lengths = batch[f'length_{mot_from}']
    # T, zeros, trans, global, local
    tensor_list = []
    trans = batch[f'body_transl_{mot_from}']
    body_pose_6d = batch[f'body_pose_{mot_from}']
    global_orient_6d = batch[f'body_orient_{mot_from}']
    trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d,
                                            trans) # T, 3, [0] is [0,0,0]
    motion_smpl = torch.cat([trans_delta, body_pose_6d,
                                    global_orient_6d], dim=-1)
    motion_smpl = normalizer(motion_smpl)
    # In some motions, translation is always zeros.
    # first 3 dimensions of the first frame should be zeros
    # B, T, D
    return motion_smpl, lengths
from src.tmr.data.motionfix_loader import Normalizer

@hydra.main(config_path="/depot/bera89/data/li5280/project/motionfix/configs", config_name="motionfix_eval")
def main(cfg: DictConfig):
    model = load_tmr_model()
    dataloader = load_testloader(cfg)
    normalizer = Normalizer("/depot/bera89/data/li5280/project/motionfix/eval-deps/stats/humanml3d/amass_feats")

    # STEP 3: dataset to SMPL
    for batch in dataloader:
        source_smpl, lengths = batch_to_smpl(batch, normalizer) # B, T, 135
        masks = length_to_mask(lengths, device=source_smpl.device)
        in_batch = {'x': source_smpl, 'mask': masks}
        res = model.encode_motion(in_batch)
        print(res.size())
        break
# what we want: original SMPL data, and lengths
# Then will converted model-specific representation

if __name__ == '__main__':
    main()
