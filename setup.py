
from setuptools import setup, find_packages
from Cython.Build import cythonize


setup(
    name='text-reuse-retrieve',
    version='0.1.18',
    description='A Python package to perform Text Reuse Detection',
    url='https://github.com/emanjavacas/retrieve',
    author='Enrique Manjavacas',
    license='MIT',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    # ext_modules=cythonize('ext/_align.pyx'), # disabling cython extension for now
    install_requires=[
        'Cython>=0.29.19',
        'numpy>=1.17.5',
        'scipy>=1.4.1',
        'gensim>=3.8.3',
        'tqdm>=4.46.0',
        'numba>=0.49.1',
        'pandas>=1.0.4',
        'scikit_learn>=0.23.2',
        'matplotlib>=3.2.1',
        'Flask>=1.1.2',
        'SetSimilaritySearch>=0.1.7'
        'dataclasses>=0.6',
        'importlib-metadata >= 1.0 ; python_version < "3.8"'
        # 'pyemd>=0.5.1',
        # 'fastText==0.9.2',
    ],
    extras_require={":python_version<'3.7'": ["importlib_resources"]},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)
