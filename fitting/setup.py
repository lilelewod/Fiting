# setup.py
from setuptools import setup, find_packages

setup(
    name='fitting',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'cloudpickle',
        'easydict',
        'numpy',
        'plyfile',
        'PyYAML',
        'scikit-learn',
        'torch',
    ],
)
