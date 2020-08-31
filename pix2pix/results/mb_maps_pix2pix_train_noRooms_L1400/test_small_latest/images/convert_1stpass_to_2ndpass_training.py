
# Put this script in the 'images' folder

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

#for root, dirs, files in os.walk('A/train/'):
#for root, dirs, files in os.walk('B/train/'):
#    for f in files:
#        new_f = f.split('_')
#        new_f.pop()
#        new_f.pop()
#        new_f = '_'.join(new_f)
#        new_f += '.png'
#        shutil.move(os.path.join(root, f), os.path.join(root, new_f))


