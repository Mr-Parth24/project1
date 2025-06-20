"""
Setup script for Agricultural SLAM System v2.0
Enhanced Visual SLAM for Agricultural Equipment Tracking
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Agricultural SLAM System - Enhanced Visual SLAM for Agricultural Applications"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "numpy>=1.21.0",
            "opencv-python>=4.8.0", 
            "PyQt6>=6.5.0",
            "pyrealsense2>=2.54.0",
            "psutil>=5.9.0",
            "PyYAML>=6.0"
        ]

setup(
    name="agricultural-slam-system",
    version="2.0.0",
    author="Mr-Parth24",
    author_email="", # Add your email if desired
    description="Enhanced Visual SLAM System for Agricultural Equipment Tracking",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mr-Parth24/project1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": ["PyOpenGL>=3.1.6", "PyOpenGL-accelerate>=3.1.6"],
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=4.0"],
        "full": ["GPUtil>=1.4.0", "scikit-learn>=1.1.0", "matplotlib>=3.6.0"]
    },
    entry_points={
        "console_scripts": [
            "agricultural-slam=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "README.md", "requirements.txt"],
    },
    keywords="slam, visual-odometry, agricultural, computer-vision, robotics, intel-realsense",
    project_urls={
        "Bug Reports": "https://github.com/Mr-Parth24/project1/issues",
        "Source": "https://github.com/Mr-Parth24/project1",
        "Documentation": "https://github.com/Mr-Parth24/project1#readme",
    },
)