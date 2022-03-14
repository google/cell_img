"""Setup file for Cell Imaging package."""

import setuptools

PY_MODULES = [
    'data_utils',
]

setuptools.setup(
    name='cell_img',
    version='0.0.1',
    description='Cell Imaging package.',
    install_requires=[
        'absl-py>=0.11.0',
        'apache-beam[gcp]>=2.34.0',
        'fsspec',
        'numpy>=1.19.5',
        'Pillow>=8.1.0',
        'pandas>=1.2.1',
        'tensorstore>=0.1.10'
    ],
    py_modules=PY_MODULES,
    packages=setuptools.find_packages(),
)
