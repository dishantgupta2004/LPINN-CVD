"""
Setup script for CVD-PINN package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cvd_pinn",
    version="0.1.0",
    author="Dishant Gupta",
    author_email="22bph016@nith.ac.in",
    description="Physics-Informed Neural Networks for Chemical Vapor Deposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dishantgupta2004/LPINN-CVD",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.5.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
        "pyyaml>=5.4.0",
        "streamlit>=1.10.0",
    ],
)