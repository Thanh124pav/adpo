from setuptools import setup, find_packages

setup(
    name="adpo",
    version="0.1.0",
    description="Advantage Decomposition Policy Optimization",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "verl>=0.2.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "pandas>=2.0.0",
        "hydra-core>=1.3.0",
        "vllm>=0.4.0",
        "wandb>=0.16.0",
    ],
)
