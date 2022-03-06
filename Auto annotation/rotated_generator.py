from argparse import ArgumentParser
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-i', '--input_dir', required=True, help='Path to images directory')
ap.add_argument('-o', '--output_dir', required=False, help='Images directory after rotation')
ap.add_argument(
    '-d',
    '--direction',
    required = False,
    default = 'both',
    choices = ['+90', '-90', 'both'],
    help = 'Right angle direction to rotate'
)
args = vars(ap.parse_args())
# Example: python rotated_generator.py -i "../Dataset/Tale of Kieu version 1866"

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, args['input_dir']).strip('/')
output_dir = os.path.join(script_dir, args['output_dir'] or input_dir + ' - Rotate')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Created folder:', output_dir)

for file_name in os.listdir(input_dir):
    if not file_name.endswith(('.jpg', '.png', 'jpeg')): continue
    image_name = os.path.splitext(file_name)[0]
    raw_image = cv2.imread(os.path.join(input_dir, file_name))

    if args['direction'] == '+90' or args['direction'] == 'both':
        image = cv2.rotate(raw_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(output_dir, image_name + '+90.jpg'), image)
        print('Saved +90deg rotated images of', file_name, 'to', output_dir)

    if args['direction'] == '-90' or args['direction'] == 'both':
        image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(output_dir, image_name + '-90.jpg'), image)
        print('Saved -90deg rotated images of', file_name, 'to', output_dir)
