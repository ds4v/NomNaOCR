'''
Implementation of patches splitting:
- Author: Nguyen Duc Duy Anh
- GitHub: https://github.com/duyanh1909
'''
import os
import glob
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from IHRNomDB_Rs import calculate_r_scores, print_intersection
from utils import read_patches, split_patches

PATH_LABELS = glob.glob("/tmp/crop/Patches/*/Transcription.txt")
PATH_SOURCE = '/tmp/crop/Patches'
PATH_TARGET = '/tmp/Patches'

# Import patches       
dataset = []
for path in PATH_LABELS: 
    dataset.extend(read_patches(path))
print(max([len(x[1]) for x in dataset])) # 24      

# Split patches into train & val DataFrames using IHR-NomDB R-Score       
r_scores = calculate_r_scores(dataset)
r_scores_sorted = sorted(r_scores, key=lambda x: x[2], reverse=True)
print(r_scores_sorted[:10)
''' Output:
[
    'DVSKTT-4 Ban ky thuc luc/DVSKTT_ban_thuc_XIV_53a_4.jpg',
    '門公以衛士不義遂殺之十二月初一日帝飲',
    1449896006
]
'''

df = pd.DataFrame(data=np.array(r_scores_sorted)[:, :2], columns=['path', 'text'])
df = df[df['path'] != 'DVSKTT-5 Ban ky tuc bien/DVSKTT_ban_tuc_XIX_23a_7.jpg']
df_train = df.sample(frac=0.8)
df_val = df.drop(df_train.index)

print_intersection(df_val['text'].tolist(), df_train['text'].tolist())
''' Output:
Characters intersection train 93.2405165456013
Characters intersection val 64.41315862838026
'''
print(len(df)) # 38318

# Apply splitting on the real data based on above DataFrames
label_train = [f'{path}\t{text}' for path, text in df_train.values]
with open(os.path.join(PATH_TARGET, 'Train.txt'), 'w+', encoding='utf-8') as f:
    f.write('\n'.join(label_train))

label_val = [f'{path}\t{text}' for path, text in df_val.values]
with open(os.path.join(PATH_TARGET, 'Validate.txt'), 'w+', encoding='utf-8') as f:
    f.write('\n'.join(label_val))
    
split_patches(df_train['path'])
split_patches(df_val['path'])
