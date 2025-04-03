from setuptools import setup, find_packages

setup(
    name="kie_invoice",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "dgl>=1.0.0",
        "numpy>=1.19.5",
        "tqdm>=4.64.0",
        "huggingface_hub>=0.10.0",
        "gdown>=4.5.1",
        "pandas>=1.3.5",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.2",
        "seaborn>=0.11.2",
    ],
)
