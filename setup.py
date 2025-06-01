from setuptools import setup, find_packages

setup(
    name="ts-ml-framework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'matplotlib'
    ]
)