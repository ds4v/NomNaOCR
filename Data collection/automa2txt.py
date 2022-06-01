from argparse import ArgumentParser, FileType
import json
import os
import re

ap = ArgumentParser()
ap.add_argument(
    '--infile', 
    required = True,
    help = 'JSON file to be processed',
    type = FileType('r', encoding='utf-8')
)
args = vars(ap.parse_args())
# Example: python automa2txt.py --infile "Luc Van Tien/automa.json"

out_dir = os.path.dirname(args['infile'].name)
data = json.load(args['infile'])
base_url = 'http://www.nomfoundation.org'
# vocabs = []

url_path = os.path.join(out_dir, 'url.txt')
nom_path = os.path.join(out_dir, 'nom.txt')
modern_path = os.path.join(out_dir, 'modern.txt')
# vocabs_path = os.path.join(out_dir, 'vocabs.txt')

with open(url_path, 'w', encoding='utf-8') as url_file, \
     open(nom_path, 'w', encoding='utf-8') as nom_file, \
     open(modern_path, 'w', encoding='utf-8') as modern_file: 
     # open(vocabs_path, 'w', encoding='utf-8') as vocabs_file:

    for page in data:
        url_file.write(base_url + page['url'] + '\n')

        # Remove line numbers and double new lines
        text = re.sub(r'\n(\n|[0-9 ]+)', '\n', page['text']) 

        # Remove position notes. Example: [1a*1*1]
        text = re.sub(r'\. \[[0-9]+[ab]\*[0-9]+\*[0-9]+\]', '', text) 

        # Unknown characters represented by ['[?]', '?', '-'] on the website
        # Note: this not remove the '?' characters at the of a sentence or a quote
        # text = re.sub(r'(\[\?\]|\?(?!\n|"))', '-', text)
        text = text.split('\n')

        for idx, sentence in enumerate(text):
            sentence = sentence.strip()
            if sentence in ['', '.']: continue

            if idx % 2 == 0: 
                sentence = re.sub(r'[!"#$%&()*+,.:;<=>@[\]^_`{|}~/\\\' ]', '', sentence)
                sentence = re.sub(r'(matchu|vech0075|-)', '?', sentence)
                nom_file.write(sentence + '\n')
                # vocabs.extend(list(sentence))
            else: 
                sentence = sentence.replace('mat-chu', '[UNK]')
                modern_file.write(sentence + '\n')

    # vocabs = set(vocabs) # remove duplicates in vocabs
    # vocabs.discard('-') # remove '-' from vocabs
    # vocabs_file.write('\n'.join(vocabs))
