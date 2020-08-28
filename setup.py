# Copyright (c) 2020 Maidbot All rights reserved.

__author__ = "donotsendmemail@maidbot.com (Maidbot)"

from setuptools import setup

with open('requirements.txt') as rqs:
    requirements = rqs.read().splitlines()

setup(
    name='pix2pix',
    version='1.0.0',
    description='Pix2Pix package for training, and using GANs',
    license='Apache2.0',
    author='Maidbot',
    python_requires='>=3.8',
    packages=[
        'pix2pix',
        'pix2pix.util',
        'pix2pix.scripts',
        'pix2pix.results',
        'pix2pix.options',
        'pix2pix.models',
        'pix2pix.mb_maps',
        'pix2pix.imgs',
        'pix2pix.docs',
        'pix2pix.datasets',
        'pix2pix.data',
    ],
    install_requires=requirements,
    url='git@github.com:Maidbot/pytorch-CycleGAN-and-pix2pix.git',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
