"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess == 'scale_maintain_ratio_and_crop':
        crop_size = 256 # NOTE: hardcoded due to the way this code was set up ... tries to get_params() before figuring out actual image size ... duplicated code ...
        if w >= crop_size and h >= crop_size:
            pass
        elif w < crop_size and h < crop_size:
            if w/crop_size < h/crop_size:
                new_w = crop_size
                new_h = int((crop_size / w) * h)
            else:
                new_w = int((crop_size / h) * w)
                new_h = crop_size
        elif w < crop_size:
            new_w = crop_size
            new_h = int((crop_size / w) * h)
        elif h < crop_size:
            new_w = int((crop_size / h) * w)
            new_h = crop_size

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    # Probability of horizontally flipping image or not
    flip_horizontally = random.random() > 0.5

    # Probability of vertically flipping image or not
    flip_vertically = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip_horizontally': flip_horizontally, 'flip_vertically': flip_vertically}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'scale_maintain_ratio' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_maintain_ratio(img, opt.crop_size, method)))
    elif 'scale_nearest256' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_nearest256(img, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip_horizontally:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip_horizontally']: # Sometimes true, sometimes false
            transform_list.append(transforms.Lambda(lambda img: __flip_horizontally(img, params['flip_horizontally'])))

    if not opt.no_flip_vertically:
        if params is None:
            transform_list.append(transforms.RandomVerticalFlip())
        elif params['flip_vertically']: # Sometimes true, sometimes false
            transform_list.append(transforms.Lambda(lambda img: __flip_vertically(img, params['flip_vertically'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __scale_maintain_ratio(img, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow >= crop_size and oh >= crop_size:
        return img
    elif ow < crop_size and oh < crop_size:
        if ow/crop_size < oh/crop_size:
            w = crop_size
            h = int((crop_size / ow) * oh)
        else:
            w = int((crop_size / oh) * ow)
            h = crop_size
    elif ow < crop_size:
        w = crop_size
        h = int((crop_size / ow) * oh)
    elif oh < crop_size:
        w = int((crop_size / oh) * ow)
        h = crop_size
    return img.resize((w, h), method)


def __scale_nearest256(img, method=Image.BICUBIC):
    ow, oh = img.size
    w = 256 * round(ow / 256)
    h = 256 * round(oh / 256)
    if w == 0:
        w = 256
    if h == 0:
        h = 256
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip_horizontally(img, flip_horizontally):
    if flip_horizontally:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __flip_vertically(img, flip_vertically):
    if flip_vertically:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
