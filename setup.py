"""Setup file for Cell Imaging package."""

import setuptools

setuptools.setup(
    name='cell_img',
    version='0.0.1',
    description='Cell Imaging package.',
    install_requires=[
        'absl-py>=0.11.0',
        'apache-beam[gcp]>=2.34.0',
        'fsspec[gcs]',
        'IPython>=7.9.0',
        'jax>=0.3.25',
        'numpy>=1.19.5',
        'pandas>=1.2.1',
        'patsy>=0.5.3',
        'Pillow>=8.1.0',
        'scipy>=1.10',
        'tensorstore>=0.1.10'
    ],
    packages=setuptools.find_packages(),
)
