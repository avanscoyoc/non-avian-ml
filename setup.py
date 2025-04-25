from setuptools import setup, find_packages

setup(
    name="non-avian-ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'torchvision',
        'numpy',
        'pandas',
        'scikit-learn',
        'librosa'
    ],
    entry_points={
        'console_scripts': [
            'evaluate-model=test.evaluate:main',
        ],
    }
)