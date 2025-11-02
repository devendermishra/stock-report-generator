"""
Setup script for Stock Report Generator.
Enables pip installation: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="stock-report-generator",
    version="1.0.0",
    description="A sophisticated multi-agent AI system for generating comprehensive equity research reports for NSE stocks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Systems Engineer",
    author_email="",
    url="https://github.com/devendermishra/stock-report-generator",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "gpu": [
            line.strip()
            for line in (Path(__file__).parent / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ] if (Path(__file__).parent / "requirements-gpu.txt").exists() else [],
        "minimal": [
            line.strip()
            for line in (Path(__file__).parent / "requirements-minimal.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ] if (Path(__file__).parent / "requirements-minimal.txt").exists() else [],
    },
    entry_points={
        "console_scripts": [
            "stock-report=src.main:cli_main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="stock, research, report, nse, finance, ai, multi-agent, langchain, langgraph",
    project_urls={
        "Bug Reports": "https://github.com/devendermishra/stock-report-generator/issues",
        "Source": "https://github.com/devendermishra/stock-report-generator",
        "Documentation": "https://github.com/devendermishra/stock-report-generator/blob/main/README.md",
    },
)
