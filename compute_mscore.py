import torch
import numpy as np
import os
from tqdm import tqdm
def load_fix_motion(motion_path):
    dic = np.load(motion_path, allow_pickle=True).item()
    motion_6d = torch.from_numpy(dic['pose']) # T, 135
    padded_global = torch.cat([motion_6d[:,:3], torch.zeros_like(motion_6d[:,:3])], dim=1)# t, 6
    reshaped_local = motion_6d[:,3:].reshape(-1, 22, 6)
    iden_6d = torch.tensor([1,0,0,0,1,0])[None,None].repeat(reshaped_local.size(0), 2,1) # T, 2, 6
    padded_local = torch.cat([reshaped_local, iden_6d], dim=1)# T, 24, 6
    motion_reshaped = torch.cat([padded_local, padded_global[:, None]], dim=1)# T, 25, 6
    motion_reshaped = motion_reshaped.permute(1,2,0)[None]# 1, 25, 6, T
    # out: [(poses, (global, 0))]
    return motion_reshaped

def evaluate_fixed_size(motion):
    # 1, 25, 6, 60
    # motion pre-processing
    preprocessed_motion = into_critic(example_motion) # [1, frame, 25, 3], axis-angle with 24 SMPL joints and 1 XYZ root location
    # critic score
    critic_scores = critic_model.module.batch_critic(preprocessed_motion)
    return critic_scores.cpu().item()

def evaluate_motion(example_motion):
    # # 1, 25, 6, T
    T = example_motion.size(-1)
    if T>60:
        example_start = example_motion[...,:60]
        example_end = example_motion[...,-60:]# 1, 25, 6, 60
        critic_1 = evaluate_fixed_size(example_start)
        example_end[:, -1, :3, :] -= example_end[:, -1, :3, :1]# last joint is global translation
        critic_2 = evaluate_fixed_size(example_end)
        return (critic_1 + critic_2)/2
    else:
        print(f"too short: {T}")
        return 0

import os

def find_npy_files(directory):
    # List to store the npy file paths
    npy_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                # Add the full path of npy file to the list
                npy_files.append(os.path.join(root, file))
    
    return npy_files


from lib.model.load_critic import load_critic
from parsedata import into_critic
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model = load_critic("pretrained/motioncritic_pre.pth", device)

# Example usage: replace 'your_directory_path' with the actual directory
directory_path = 'PATH/TO/NPY/FILES'
npy_files = find_npy_files(directory_path)

score_list = []
for file in tqdm(npy_files):
    example_motion = load_fix_motion(motion_path=file).to(device)
    score = evaluate_motion(example_motion)
    score_list.append(score)

print(np.mean(score_list))