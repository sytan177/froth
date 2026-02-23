from setuptools import setup, find_packages


setup(
    name="froth",
    version="0.1.0",
    packages=find_packages(),
    package_data={'froth': ['data/**/*', 'data/*']},
    include_package_data=True,
    install_requires=[
        "numpy>=1.26",
        "torch>=2.7",
        "torch-geometric>=2.6",
        "torch-scatter",
        "torch-sparse",
        "scikit-learn>=1.6",
        "h5py>=3.11",
        "numba>=0.60",
        "joblib>=1.4",
        "photutils>=1.11",
    ],
    python_requires='>=3.9',
)
