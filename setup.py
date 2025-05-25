from setuptools import setup, find_packages

setup(
    name='ts-ml-framework',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A basic end-to-end machine learning framework for time series prediction.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'pandas',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)