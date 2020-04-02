
# Put this script in the 'images' folder

# 1. Run this script to get images sorted into A/ and B/
# 2. Run the temp_combine.sh script to combine images from fold_A and fold_B into a single paired image dataset

import os
import shutil

try:
    os.makedirs('A/train')
    os.makedirs('B/train')
except:
    pass

for root, dirs, files in os.walk('./'):
    for f in files:
        new_filename = f.split('_')
        new_filename.pop()
        new_filename.pop()
        new_filename = '_'.join(new_filename)
        new_filename += '.png'
        if 'fake_B' in f:
            shutil.copyfile(f, f'A/train/{new_filename}')
        elif 'real_B' in f:
            shutil.copyfile(f, f'B/train/{new_filename}')

