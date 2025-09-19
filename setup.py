"""
Setup script for Hawkes Process Causality Detection
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hawkes-process-causality",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python implementation for learning Granger causality in Hawkes processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hawkes-process-causality",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="hawkes process, granger causality, point process, machine learning, time series",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hawkes-process-causality/issues",
        "Source": "https://github.com/yourusername/hawkes-process-causality",
        "Documentation": "https://github.com/yourusername/hawkes-process-causality/blob/main/README.md",
    },
)