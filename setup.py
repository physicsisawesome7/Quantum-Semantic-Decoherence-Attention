"""
setup.py — makes `pip install -e .` work so imports resolve cleanly.
"""
from setuptools import setup, find_packages

setup(
    name="qsda",
    version="2.0.0",
    description="Quantum Semantic Decoherence Attention — uncertainty-aware language modelling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="QSDA Research",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "pytest-cov>=4.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
