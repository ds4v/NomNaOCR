from argparse import ArgumentParser, FileType
import json
import os

ap = ArgumentParser()
ap.add_argument(
    '--infile', 
    required = True,
    help = 'JSON file to be processed',
    type = FileType('r', encoding='utf-8')
)
args = vars(ap.parse_args())
# Example: python json2txt.py --infile "Tale of Kieu version 1866/automa.json"

data = json.load(args['infile'])
base_url = 'http://www.nomfoundation.org'
out_dir = os.path.dirname(args['infile'].name)

url_path = os.path.join(out_dir, 'url.txt')
nom_path = os.path.join(out_dir, 'nom.txt')
modern_path = os.path.join(out_dir, 'modern.txt')

with open(url_path, 'w', encoding='utf-8') as url_file:
    with open(nom_path, 'w', encoding='utf-8') as nom_file:
        with open(modern_path, 'w', encoding='utf-8') as modern_file:
            for page in data:
                url_file.write(base_url + page['url'] + '\n')
                text = page['text'].replace('\n\n', '\n').split('\n')

                for idx, sentence in enumerate(text):
                    if sentence == '': continue
                    if idx % 2 == 0: nom_file.write(sentence + '\n')
                    else: modern_file.write(sentence + '\n')

