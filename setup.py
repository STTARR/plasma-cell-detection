from setuptools import find_packages, setup

setup(
    name='mmdetect',
    packages=find_packages(),
    version='0.1.0',
    description='Automated plasma cell detection using deep learning.',
    author='Fred Fu (STTARR Innovation Centre)',
    # Packages should be installed with conda (because torch requires CUDA), but
    # this will check that they exist in the current environment at least.
    install_requires=[
        'numpy',
        'pandas',
        'pillow',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'scikit-image',
        'torch',
        'torchvision',
        'flask',
    ]
)
