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
out_name = os.path.basename(args['infile'].name)
out_name = os.path.splitext(out_name)[0] + '.txt'

data = json.load(args['infile'])
with open(out_name, 'w', encoding='utf-8') as file:
    for page in data:
        page = page['text'].split('\n')
        for idx, sentence in enumerate(page):
            if idx % 2 == 0 and sentence != '':
                file.write(sentence + '\n')
