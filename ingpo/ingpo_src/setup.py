from setuptools import find_packages, setup

setup(
    name="ingpo_ext",
    version="0.1.0",
    description="InGPO: Information-Gated Policy Optimization extension for SPO",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sortedcontainers",
        "httpx",
        "openai>=1.0",
    ],
    python_requires=">=3.9",
)
