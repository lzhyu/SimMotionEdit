import os
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict
from torch import Tensor

from src.utils.file_io import write_json, read_json
from tmr_evaluator.fid import evaluate_fid
logger = logging.getLogger(__name__)

mat2name = {
            'sim_matrix_s_t': 'source_target',
            'sim_matrix_t_t': 'target_generated'
            }

import os
import json
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig) -> str:
    path = os.path.join(cfg.run_dir, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        string = json.dumps(config, indent=4)
        f.write(string)
    return path


def read_config(run_dir: str, return_json=False) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg


def length_to_mask(length, device: torch.device = None, max_len=None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)
    if max_len is not None:
        max_len = max_len
    else:
        max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def line2dict(line):
    names_of_metrics = ["R@1_s2t", "R@2_s2t", "R@3_s2t", "R@5_s2t", "R@10_s2t", "MedR_s2t", "AvgR_s2t",
                        "R@1", "R@2", "R@3", "R@5", "R@10", "MedR", "AvgR"]
    metrics_nos = line.replace('\\', '').split('&')
    metrics_nos = [x.strip() for x in metrics_nos if x]
    return dict(zip(names_of_metrics, metrics_nos))

def lengths_to_mask_njoints(lengths: List[int], njoints: int, device: torch.device) -> Tensor:
    # joints*lenghts
    joints_lengths = [njoints*l for l in lengths]
    joints_mask = lengths_to_mask(joints_lengths, device)
    return joints_mask


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len,
                        device=device).expand(len(lengths),
                                              max_len) < lengths.unsqueeze(1)
    return mask

def collect_gen_samples(gener_motions, normalizer, device):
    cur_samples = {}
    cur_samples_raw = {}
    # it becomes from 
    # translation | root_orient | rots --> trans | rots | root_orient 
    from src.data.features import _get_body_transl_delta_pelv_infer
    # NOTE: changing the represetation
    if isinstance(gener_motions, str):
        # you have a path and not the motions themselves
        import glob
        sample_files = glob.glob(f'{gener_motions}/*.npy')
        for fname in tqdm(sample_files):
            keyid = str(Path(fname).name).replace('.npy', '')
            gen_motion_b = np.load(fname,
                                allow_pickle=True).item()['pose']
            gen_motion_b = torch.from_numpy(gen_motion_b)
            trans = gen_motion_b[..., :3]
            global_orient_6d = gen_motion_b[..., 3:9]
            body_pose_6d = gen_motion_b[..., 9:]
            trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d,
                                                    trans) # T, 3, [0] is [0,0,0]
            gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                            global_orient_6d], dim=-1)
            gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
            cur_samples[keyid] = gen_motion_b_fixed.to(device)
            cur_samples_raw[keyid] = torch.cat([trans, global_orient_6d,
                                                body_pose_6d], dim=-1).to(device)
            # print('gen motion')
            # print(gen_motion_b_fixed[:10])
            # print(gen_motion_b_fixed.shape)
    else:
        for keyid, motion_feats in gener_motions.items():
            trans = motion_feats[..., :3]
            global_orient_6d = motion_feats[..., 3:9]
            body_pose_6d = motion_feats[..., 9:]
            trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d,
                                                    trans)
            gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                            global_orient_6d], dim=-1)
            gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
            cur_samples[keyid] = gen_motion_b_fixed.to(device)
            cur_samples_raw[keyid] = torch.cat([trans, global_orient_6d, 
                                                body_pose_6d], dim=-1).to(device)

    return cur_samples, cur_samples_raw

def compute_sim_matrix(model, dataset, keyids, gen_samples,
                       batch_size=256, progress=True):
    import torch
    import numpy as np
    from src.data.tools.collate import collate_text_motion
    from src.tmr.tmr import get_sim_matrix
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    keyids_ordered = {}
    with torch.no_grad():

        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)
        # by batch (can be too costly on cuda device otherwise)
        for sett in ['s_t', 't_t']:
            cur_samples = []
            latent_motions_A = []
            latent_motions_B = []
            keys_ordered_for_run = []

            if progress:
                data_iter = tqdm(all_data_splitted, leave=False)
            else:
                data_iter = all_data_splitted
            for data in data_iter:
                # batch = collate_text_motion(data, device=device)
                from src.data.tools.collate import collate_tensor_with_padding
                cur_batch_keys = [x['keyid'] for x in data]
                keys_ordered_for_run.extend(cur_batch_keys)
                # TODO load the motions for the generations
                # Text is already encoded
                if sett == 's_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    lengths_tgt = [len(x['motion_target']) for x in data]
                    if gen_samples:
                        cur_samples = [gen_samples[key_in_batch][:lengths_tgt[ix]] for ix, key_in_batch in enumerate(cur_batch_keys)]
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(
                            cur_samples).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding(
                           [x['motion_target'] for x in data]).to(model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                    masks_a = length_to_mask(lengths_a, device=motion_a.device)
                    masks_b = length_to_mask(lengths_b, device=motion_b.device)
                    motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                    'x': motion_a}
                    motion_b_dict = {'length': lengths_b, 'mask': masks_b, 
                                    'x': motion_b}
                elif sett == 't_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_target']) for x in data]
                    lengths_tgt = [len(x['motion_target']) for x in data]

                    if gen_samples:
                        cur_samples = [gen_samples[key_in_batch][:lengths_tgt[ix]] for ix, key_in_batch in enumerate(cur_batch_keys)]
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(cur_samples
                                                               ).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding([
                            x['motion_target'] for x in data]).to(
                                model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                    masks_a = length_to_mask(lengths_a, device=motion_a.device)
                    masks_b = length_to_mask(lengths_b, device=motion_b.device)
                    motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                    'x': motion_a}
                    motion_b_dict = {'length': lengths_b, 'mask': masks_b, 
                                    'x': motion_b}

                # Encode both motion and text
                latent_motion_A = model.encode(motion_a_dict, 
                                            sample_mean=True)
                latent_motion_B = model.encode(motion_b_dict,
                                            sample_mean=True)
                latent_motions_A.append(latent_motion_A)
                latent_motions_B.append(latent_motion_B)

            latent_motions_A = torch.cat(latent_motions_A)
            latent_motions_B = torch.cat(latent_motions_B)
            sim_matrix = get_sim_matrix(latent_motions_A, latent_motions_B)
            returned[f'sim_matrix_{sett}'] = sim_matrix.cpu().numpy()
            keyids_ordered[sett] = keys_ordered_for_run
    return returned, keyids_ordered

def get_motion_distances(model, dataset, keyids, gen_samples,
                         batch_size=256):

    import torch
    import numpy as np
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    print(len(dataset))
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    import smplx
    body_model = smplx.SMPLHLayer(f'data/body_models/smplh',
                                    model_type='smplh',
                                    gender='neutral',
                                    ext='npz').to('cuda').eval()

    with torch.no_grad():
        all_data = [dataset.load_keyid_raw(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)
        # by batch (can be too costly on cuda device otherwise)
        for sett in ['t_t']:
            cur_samples = []
            motions_a = []
            motions_b = []
            tot_lens_a = []
            tot_lens_b = []
            for data in tqdm(all_data_splitted, leave=False):
                # batch = collate_text_motion(data, device=device)
                from src.data.tools.collate import collate_tensor_with_padding
                # TODO load the motions for the generations
                keyids_of_cursplit = [x['keyid'] for x in data]
                # Text is already encoded
                if sett == 's_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    if gen_samples:
                        cur_samples = [gen_samples[kd] for kd in keyids_of_cursplit]
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(
                            cur_samples).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding(
                           [x['motion_target'] for x in data]).to(model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                elif sett == 't_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_target']) for x in data]
                    if gen_samples:
                        cur_samples = [gen_samples[kd] for kd in keyids_of_cursplit]
                        lengths_b = [len(x) for x in cur_samples]
                        if motion_a.shape[1] < cur_samples[0].shape[0]:
                            cur_samples = [cs[:motion_a.shape[1]] for cs in cur_samples]
                            motion_b = collate_tensor_with_padding(cur_samples
                                                                ).to(model.device)
                        else:
                            motion_b = collate_tensor_with_padding(cur_samples
                                                                ).to(model.device)
                            
                    else:
                        motion_b = collate_tensor_with_padding([
                            x['motion_target'] for x in data]).to(
                                model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                def split_into_chunks(N, k): 
                    chunked = [k*i for i in range(1, N//k+1)] + ([N] if N%k else [])
                    return [0] + chunked

                ids_for_smpl = split_into_chunks(motion_a.shape[0], 16)
                def sliding_window(lst):
                    return [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]

                for s, e in sliding_window(ids_for_smpl):
                    motions_a.append(run_smpl_fwd(motion_a[s:e, :, :3],
                                                motion_a[s:e, :, 3:9],
                                                motion_a[s:e, :, 9:],
                                                body_model).detach().cpu())
                    motions_b.append(run_smpl_fwd(motion_b[s:e, :, :3],
                                                motion_b[s:e, :, 3:9],
                                                motion_b[s:e, :, 9:],
                                                body_model).detach().cpu())
                tot_lens_a.extend(lengths_a)
                tot_lens_b.extend(lengths_b)

            mask_a = lengths_to_mask(tot_lens_a, device).detach().cpu()
            mask_b = lengths_to_mask(tot_lens_b, device).detach().cpu()

            from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
            max_a = -5
            for x in motions_a:
                if len(x[0]) > max_a:
                    max_a = len(x[0])
            max_b = -5
            for x in motions_b:
                if len(x[0]) > max_b:
                    max_b = len(x[0])

            motions_a_proc = []
            for x in motions_a:
                if len(x[0]) != max_a:
                    zeros_to_add = torch.zeros(x.size(0),
                                               max_a - len(x[0]), 
                                               73, 3) # 6890 is the number of vertices
                    motions_a_proc.append(torch.cat((x, 
                                                     zeros_to_add), dim=1))
                else:
                    motions_a_proc.append(x)

            motions_b_proc = []
            for x in motions_b:
                if len(x[0]) != max_b:
                    zeros_to_add = torch.zeros(x.size(0),
                                               max_b - len(x[0]), 
                                               73, 3)
                    motions_b_proc.append(torch.cat((x, 
                                                     zeros_to_add), dim=1))
                else:
                    motions_b_proc.append(x)


            from einops import rearrange
            motions_a = torch.cat(motions_a_proc).detach().cpu() # B, T, J, 3
            motions_b = torch.cat(motions_b_proc).detach().cpu()
            # check shape
            print(motions_a.shape)
            # 100*
            common_mask = torch.logical_and(mask_a, mask_b)
            global_edit_accuracy = mse_loss(motions_a, motions_b, 
                                           reduction='none').flatten(-2,-1).mean(-1)*common_mask
            global_edit_accuracy = torch.sqrt(global_edit_accuracy)
            tot_gl_edacc = global_edit_accuracy.sum() / common_mask.sum()

            # global_edit_accuracy = global_edit_accuracy.mean()

            returned[f'distances_{sett}'] = tot_gl_edacc.cpu().numpy()

    return returned

def run_smpl_fwd(body_transl, body_orient, body_pose, body_model,
                 verts=True):
    from src.tools.transforms3d import transform_body_pose
    
    if len(body_transl.shape) > 2:
        bs, seqlen = body_transl.shape[:2]
        body_transl = body_transl.flatten(0, 1)
        body_orient = body_orient.flatten(0, 1)
        body_pose = body_pose.flatten(0, 1)

    batch_size = body_transl.shape[0]
    body_model.batch_size = batch_size
    verts = body_model(transl=body_transl, body_pose=transform_body_pose(body_pose,
                                                            '6d->rot'),
                      global_orient=transform_body_pose(body_orient,
                                                        '6d->rot')).joints # vertices
    return verts.reshape(bs, seqlen, -1, 3)

def shorten_metric_line(line_to_shorten):
    # Split the string into a list of numbers
    numbers = line_to_shorten.split('&')

    # Remove the elements at the 4th, 5th, 6th, 11th, 12th, and 13th indices
    indices_to_remove = [4, 5, 6, 11, 12, 13]
    for index in sorted(indices_to_remove, reverse=True):
        del numbers[index]

    # Join the list back into a string
    return '&'.join(numbers)

def retrieval(samples_to_eval) -> None:
    # 
    protocol = ['normal', 'batches']
    device = 'cuda'
    run_dir = 'eval-deps'
    ckpt_name = 'last'
    batch_size = 256
    
    protocols = protocol
    dataset = 'motionfix' # motionfix
    sets = 'test' # val all
    # save_dir = os.path.join(run_dir, "motionfix/contrastive_metrics")
    # os.makedirs(save_dir, exist_ok=True)

    # Load last config
    curdir = Path(hydra.utils.get_original_cwd())

    cfg = read_config(curdir / run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.tmr.load_model import load_model_from_cfg
    from src.tmr.metrics import all_contrastive_metrics_mot2mot, print_latex_metrics_m2m

    pl.seed_everything(cfg.seed)

    logger.info("Loading the evaluation TMR model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    datasets = {}
    results = {}
    keyids_ord = {}
    bs_m2m = 32 # for the batch size metric
    # calculate splits
    from src.tmr.data.motionfix_loader import Normalizer
    normalizer = Normalizer(curdir/run_dir/'stats/humanml3d/amass_feats')
    print(curdir/run_dir/'stats/humanml3d/amass_feats')
    # obtaining the generated samples
    gen_samples, gen_samples_raw = collect_gen_samples(samples_to_eval,
                                        normalizer, 
                                        model.device)
    # gen sample are used
    # NOTE: the function is the key
    results_v2v = {}
    exist_gen_keys = list(gen_samples.keys())
    if sets == 'all':
        sets_to_load = ['val', 'test']
        extra_str = '_val_test'
    elif sets == 'val':
        sets_to_load = ['val']
        extra_str = '_val'
    else:
        sets_to_load = ['test']
        extra_str = ''

    # protocols = ['batches']
    for protocol in protocols:
        # logger.info(f"|------Protocol {protocol.upper()}-----|")
        # Load the dataset if not already
        if protocol not in datasets:
            from src.tmr.data.motionfix_loader import MotionFixLoader
            dataset = MotionFixLoader(sets=sets_to_load, 
                                      keys_to_load=exist_gen_keys)

            datasets.update(
                {key: dataset for key in ["normal", "batches"]}
            )
        gen_samples = {k:v for k, v in gen_samples.items() if k in dataset.motions.keys()}
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol=="normal":
                res, keyids_ord_for_all = compute_sim_matrix(
                    model, dataset, dataset.keyids, 
                    gen_samples=gen_samples,
                    batch_size=batch_size,
                )
                keyids_ord['all'] = keyids_ord_for_all
                results.update({key: res for key in ["normal"]})

                # dists = get_motion_distances(
                #     model, dataset, dataset.keyids, 
                #     gen_samples=gen_samples_raw,
                #     batch_size=batch_size,
                # )
                # print(dists)

            elif protocol == "batches":
                # print('running batch')
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[bs_m2m * i : bs_m2m * (i + 1)] for i in range(len(keyids) // bs_m2m)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["batches"] = []
                keyids_ord["batches"] = []

                results["features_gt"] = []
                results["features_gen"] = []

                for idx_batch in tqdm(idx_batches):
                    # it reles on compute_sim_matrix
                    res_matrs, res_keys = compute_sim_matrix(model, dataset,
                                                             np.array(keyids)[idx_batch],
                                                             gen_samples=gen_samples,
                                                             batch_size=batch_size,
                                                             progress=False)
                    # TODO: compute features
                    results["batches"].append(res_matrs)
                    keyids_ord["batches"].append(res_keys)

                    res_feats, _ = compute_features(model, dataset,
                                        np.array(keyids)[idx_batch],
                                        gen_samples=gen_samples,
                                        batch_size=batch_size,
                                        progress=False)
                    
                    results["features_gt"].append(res_feats['features_gt'])
                    results["features_gen"].append(res_feats['features_gen'])

        result = results[protocol]

        # Compute the metrics
        if protocol == "batches":
            protocol_name = protocol
            def compute_batches_metrics(sim_matrix_lst):
                all_metrics = []
                all_cols = []
                for sim_matrix in sim_matrix_lst:
                    metrics, cols_for_metr = all_contrastive_metrics_mot2mot(sim_matrix,
                                                      rounding=None,  return_cols=True)
                    all_metrics.append(metrics)
                    all_cols.append(cols_for_metr)

                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = round(
                        float(np.mean([metrics[key] for metrics in all_metrics])), 2
                    )
                return avg_metrics, all_cols
            metrics_dico = {}
            result_packed_to_d = {key: [d[key] for d in result]
                                  for key in result[0]
                                  }
            keyids_ord['batches'] = {key: [d[key] for d in keyids_ord["batches"]]
                        for key in keyids_ord["batches"][0]
                        }
            str_for_tab = ''

            for var, lst_of_sim_matrs in result_packed_to_d.items():
                metr_name = mat2name[var]
                if var == 'sim_matrix_s_t':
                    keyids_for_sel = keyids_ord['batches']['s_t']
                else:
                    keyids_for_sel = keyids_ord['batches']['t_t']

                metrics_dico[metr_name], cols_for_metr_temp = compute_batches_metrics(lst_of_sim_matrs)

                idxs_good = [np.where(el < 2)[0] for el in cols_for_metr_temp] 
                cols_for_metr_unmerged = [list(np.array(for_sel_cur
                                                         )[idxs]) for idxs, for_sel_cur in zip(idxs_good, keyids_for_sel)]
                cols_for_metr[metr_name] = [item for sublist in cols_for_metr_unmerged for item in sublist]
                str_for_tab += print_latex_metrics_m2m(metrics_dico[metr_name])
                metric_name = f"{protocol_name}_{metr_name}.yaml"
            cand_keyids_batches = cols_for_metr['target_generated']
            # if motion_gen_path is not None:
            #     write_json(cand_keyids_guo, Path(motion_gen_path) / f'guo_candkeyids{extra_str}.json')
            line_for_batches = str_for_tab.replace("\\\&", "&")

        else:
            protocol_name = protocol
            emb, threshold = None, None
            metrics = {}
            cols_for_metr = {}
            str_for_tab = ''
            for var, sim_matrix in result.items():
                metr_name = mat2name[var]
                if var == 'sim_matrix_s_t':
                    keyids_for_sel = keyids_ord['all']['s_t']
                else:
                    keyids_for_sel = keyids_ord['all']['t_t']

                metrics[metr_name], cols_for_metr_temp = all_contrastive_metrics_mot2mot(sim_matrix, 
                                                    emb, threshold=threshold, return_cols=True)
                idxs_good = np.where(cols_for_metr_temp < 5)[0]
                cols_for_metr[metr_name] = list(np.array(keyids_for_sel
                                                         )[idxs_good])
                str_for_tab += print_latex_metrics_m2m(metrics[metr_name])
            cand_keyids_all = cols_for_metr['target_generated']
            line_for_all = str_for_tab.replace("\\\&", "&")
            # TODO do this at some point!
            # run = wandb.init()
            # my_table = wandb.Table(columns=["a", "b"],
            #                        data=[["1a", "1b"], ["2a", "2b"]])
            # run.log({"table_key": my_table})

    dict_batches = line2dict(line_for_batches)
    dict_full = line2dict(line_for_all)
    names_to_keep = ["R@1_s2t", "R@2_s2t", "R@3_s2t", 
                    "R@1", "R@2", "R@3", "AvgR"]
    metrs_full = {key: dict_full[key] for key in names_to_keep if key in dict_full}
    metrs_batches = {key: dict_batches[key] for key in names_to_keep if key in dict_batches}

    return metrs_batches, metrs_full

def compute_features(model, dataset, keyids, gen_samples,
                       batch_size=256, progress=True):
    import torch
    import numpy as np
    from src.data.tools.collate import collate_text_motion
    from src.tmr.tmr import get_sim_matrix
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    keyids_ordered = {}
    with torch.no_grad():

        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)
        # by batch (can be too costly on cuda device otherwise)
        sett = 't_t'
        cur_samples = []
        latent_motions_A = []
        latent_motions_B = []
        keys_ordered_for_run = []

        if progress:
            data_iter = tqdm(all_data_splitted, leave=False)
        else:
            data_iter = all_data_splitted
        for data in data_iter:
            # batch = collate_text_motion(data, device=device)
            from src.data.tools.collate import collate_tensor_with_padding
            cur_batch_keys = [x['keyid'] for x in data]
            keys_ordered_for_run.extend(cur_batch_keys)
            # TODO load the motions for the generations
            # Text is already encoded
            if sett == 't_t':
                motion_a = collate_tensor_with_padding(
                    [x['motion_target'] for x in data]).to(model.device)
                lengths_a = [len(x['motion_target']) for x in data]
                lengths_tgt = [len(x['motion_target']) for x in data]

                if gen_samples:
                    cur_samples = [gen_samples[key_in_batch][:lengths_tgt[ix]] for ix, key_in_batch in enumerate(cur_batch_keys)]
                    lengths_b = [len(x) for x in cur_samples]
                    motion_b = collate_tensor_with_padding(cur_samples
                                                            ).to(model.device)
                else:
                    motion_b = collate_tensor_with_padding([
                        x['motion_target'] for x in data]).to(
                            model.device)
                    lengths_b = [len(x['motion_target']) for x in data]

                masks_a = length_to_mask(lengths_a, device=motion_a.device)
                masks_b = length_to_mask(lengths_b, device=motion_b.device)
                motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                'x': motion_a}
                motion_b_dict = {'length': lengths_b, 'mask': masks_b, 
                                'x': motion_b}

            # Encode both motion and text
            latent_motion_A = model.encode(motion_a_dict, 
                                        sample_mean=True)
            latent_motion_B = model.encode(motion_b_dict,
                                        sample_mean=True)
            latent_norm_A = latent_motion_A.norm(p=2, dim=1, keepdim=True)
            latent_norm_B = latent_motion_B.norm(p=2, dim=1, keepdim=True)
            # 归一化
            latent_motions_A.append(latent_motion_A/ latent_norm_A)
            latent_motions_B.append(latent_motion_B/ latent_norm_B)

        latent_motions_A = torch.cat(latent_motions_A)
        latent_motions_B = torch.cat(latent_motions_B)
        returned[f"features_gt"] = latent_motions_A # B, D
        returned[f"features_gen"] = latent_motions_B
        keyids_ordered[sett] = keys_ordered_for_run
    return returned, keyids_ordered

if __name__ == "__main__":
    retrieval()
