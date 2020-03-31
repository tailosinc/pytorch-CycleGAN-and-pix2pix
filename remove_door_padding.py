
import os
import shutil
import PIL
from PIL import Image
import cv2
import numpy as np

door_padding_color = [119, 119, 119]
fake_background_grey_color = [205, 205, 205]
background_grey_color = [204, 204, 204]

# NOTE: change these as needed
color_to_replace = door_padding_color
color_to_replace_with = background_grey_color

print('files modified:')
for root, dirs, files in os.walk('./'):
    for f in files:
        if 'png' in f:
            im_np_array = cv2.imread(f)
            indices = np.where(np.all(im_np_array == color_to_replace, axis=-1))
            indices_zipped = list(zip(indices[0], indices[1]))
            for i in indices_zipped:
                im_np_array[i] = color_to_replace_with
            if len(indices_zipped) > 20: # Have to do something like this because some maps aren't perfectly clean and have some splats of random colors
                print(f)
                im = Image.fromarray(im_np_array)
                im.save(f)
