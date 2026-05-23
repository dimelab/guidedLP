#!/usr/bin/env python3
"""Setup script for the guidedLP package.

Canonical install:
    cd guidedLP
    pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="guidedLP",
    version="0.1.0",
    description="Large-scale network analysis with Guided Label Propagation for computational social science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/alterPublics/guidedLP",
    project_urls={
        "Documentation": "https://guided-label-propagation.readthedocs.io",
        "Repository": "https://github.com/alterPublics/guidedLP.git",
        "Issues": "https://github.com/alterPublics/guidedLP/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "networkit>=11.0",
        "polars>=0.20.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "viz": [
            "matplotlib>=3.6",
            "plotly>=5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Sociology",
    ],
    keywords=["network-analysis", "community-detection", "social-science", "label-propagation"],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
