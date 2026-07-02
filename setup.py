from setuptools import find_packages, setup

setup(
    name="ada-covid",
    version="1.0.0",
    description="Adversarial Deep Domain Adaptation with Triplet Loss",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8",
        "pillow",
        "scikit-learn",
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
