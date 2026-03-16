import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='openood-vlm',
    version='1.0',
    author='openood dev team',
    author_email='22040319r@connect.polyu.hk',
    description=
    'This package provides a unified test platform for Out-of-Distribution detection.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ZhuWenjie98/ANTS',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=2.8.0', 'torchvision>=0.23.0', 'scikit-learn', 'json5',
        'matplotlib', 'scipy', 'tqdm', 'pyyaml>=5.4.1', 'pre-commit',
        'opencv-python>=4.4.0.46', 'imgaug>=0.4.0', 'pandas', 'diffdist>=0.1',
        'Cython>=0.29.30', 'faiss-gpu>=1.7.2', 'gdown>=4.7.1', 'libmr>=0.1.9',
        'transformers==4.57.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
)
