'''
Implementation of pages splitting:
- Author: Nguyen Duc Duy Anh
- GitHub: https://github.com/duyanh1909
'''
import os
import glob
import shutil
from tqdm.notebook import tqdm
from utils import read_pages, split_pages


tool_label = {}
tool_label['Train'] = []
tool_label['Validate'] = []
train_labels, val_labels = [], []

for path in os.listdir('final_datasets/label_text/'):
    full = glob.glob(os.path.join('final_datasets/label_text/', path, 'label_detect', '*'))
    ratio = len(full) * 4 // 5
    train, val = full[:ratio], full[ratio:]

    for paths, status in zip([train, val], ['Train', 'Validate']):
        target_path = split_pages(paths, '/tmp/Pages')
        tool_label[status].extend(list(map(lambda x: x[0], target_path)))
        
        if status == 'Train':
            train_labels.extend([f'{img}\t{label}' for img, label in target_path])
        else:
            val_labels.extend([f'{img}\t{label}' for img, label in target_path])
            

with open(os.path.join('/tmp/Pages', f'Train.txt'), 'w+') as f:
    f.write('\n'.join(train_labels))

with open(os.path.join('/tmp/Pages', f'Validate.txt'), 'w+') as f:
    f.write('\n'.join(val_labels))
    
for label in glob.glob('/tmp/label_pages/*/*'):
    _, _, _, dir_name, _ = label.split('/')
    shutil.copy(label, os.path.join('/tmp/Pages', dir_name, 'imgs', 'Label.txt'))
    
    
# Export to PaddleOCR format
dataset = []
for path in glob.glob('final_datasets/label_text/*/Label.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        dataset.extend([read_pages(data) for data in f.readlines()])
        
train_paths = list(map(lambda x: x.replace('imgs/', ''), tool_label['Train']))
val_paths = list(map(lambda x: x.replace('imgs/', ''), tool_label['Validate']))
label_train, label_val = [], []

for data in dataset:
    list_text = []
    
    for d in data[1]:
        text = d.copy()
        if text['difficult']: text['transcription'] = '###'
        del text['difficult']
        list_text.append(text)
        
    dir_name, img_name = data[0].split('/')
    list_text = json.dumps(list_text, ensure_ascii=False).encode('utf-8').decode()
    
    if data[0] in train_paths:
        label_train.append(f'{dir_name}/imgs/{img_name}\t{list_text}')
    elif data[0] in val_paths:
        label_val.append(f'{dir_name}/imgs/{img_name}\t{list_text}')
        
# len(label_train) # 2359
# len(label_val) # 594
# len(glob.glob('/tmp/Pages/*/gts/*')) # 2953
# len(glob.glob('/tmp/Pages/PPOCRLabel/*/imgs/*')) # 2953
# len(glob.glob('/tmp/Pages/*/imgs/*.jpg')) - len(glob.glob('/tmp/Pages/PPOCRLabel/*/imgs/*')) # 0

with open(os.path.join('/tmp/Pages/PPOCRLabel', 'Train.txt'), 'w+', encoding='utf-8') as f:
    f.write('\n'.join(label_train))

with open(os.path.join('/tmp/Pages/PPOCRLabel', 'Validate.txt'), 'w+', encoding='utf-8') as f:
    f.write('\n'.join(label_val))