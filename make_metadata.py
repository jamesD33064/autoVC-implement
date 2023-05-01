import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cpu()
c_checkpoint = torch.load('3000000-BL.ckpt',map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)

# 指的是說一個語者說了幾種不同內容的話，讓資料的數量盡量一樣，內容可以不一樣。
num_uttrs = 68
len_crop = 80

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

speakers = []
for speaker in sorted(subdirList[1:]):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    fileList = fileList[:num_uttrs]
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        # print(tmp.shape)
        # pad if the current one is too short   
        if tmp.shape[1] <= len_crop: # 過短要填充
            pad = int(len_crop - tmp.shape[1]) #要填充多少
            tmp = pad_along_axis(pad,tmp)
            melsp = torch.from_numpy(tmp[np.newaxis,:, :]).cpu()
        else:              
            left = np.random.randint(0, tmp.shape[1]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis,:, left:left+len_crop]).cpu()
        # print(melsp.shape)
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())    
        
utterances.append(np.mean(embs, axis=0))
for fileName in sorted(fileList):
    utterances.append(os.path.join(speaker,fileName))
speakers.append(utterances)

with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)