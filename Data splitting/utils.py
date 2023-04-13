import os
import re
import ast
import shutil
from tqdm.notebook import tqdm


def read_patches(path):
    path_file = re.findall(r'Patches\/(.*)?\/', path)[0]
    with open(path, 'r', encoding='utf-8') as f:
        
        
def read_pages(text):
    url, list_dict = text.split('\t')
    list_dict = ast.literal_eval(
        list_dict.replace('\n', '')
        .replace('false', 'False')
        .replace('true', 'True')
    )
    return [url, list_dict]


def split_pages(source_paths, target):
    folder_imgs = 'imgs'
    folder_labels = 'gts'
    target_path = []
    
    for source_label in tqdm(source_paths):
        dir_name = re.findall(r'.*\/label_text\/(.*)\/label_?', source_label)[0]
        label_name = os.path.basename(source_label)
        img_name = label_name.replace('txt', 'jpg')

        source_img = os.path.join('/tmp', dir_name, img_name)
        target_label = os.path.join(target, dir_name, folder_labels, label_name)
        target_img = os.path.join(target, dir_name, folder_imgs, img_name)
        tool_img = os.path.join(target, 'PPOCRLabel', dir_name, folder_imgs, img_name)

        shutil.copy(source_label, target_label)
        shutil.copy(source_img, target_img)
        shutil.copy(source_img, tool_img)

        target_path.append([
            target_img.replace('/tmp/Pages/', ''), 
            target_label.replace('/tmp/Pages/', '')
        ])
    return target_path


def split_patches(labels):
    for elm in tqdm(labels):
        path = elm.split('\t')[0]
        source = os.path.join(PATH_SOURCE, path)
        source = source.replace('crop_img/', '')

        target = path.replace('crop_img/', '')
        target = target.replace('_crop', '')
        target = os.path.join(PATH_TARGET, target)
        shutil.copy(source, target)
        
        
def check_pages_quality():
    total, diff = 0, 0
    for path in glob.glob('/tmp/Pages/*/gts/*'):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total += len(lines)
            for data in lines:
                if '###' in data: diff += 1
    return total, diff # 38613, 295