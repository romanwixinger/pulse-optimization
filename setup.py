from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pulse-opt",
    version="0.1.0",
    description="Pulse Optimization with the Noisy Quantum Gates Approach.",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="",
    author="R. Wixinger, G. D. Bartolomeo, M. Vischi, M. Grossi",
    packages=find_packages(exclude=["scripts"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5",
        "numpy>=1.21",
        "pandas>=1.4.0",
        "scipy>=1.9",
        "quantum_gates>=1.0.3",
        "qiskit-aer==0.12.0",
        "qiskit==0.42.1",
        "qiskit-ibm-provider==0.4.0",
        "qiskit-ibmq-provider==0.20.2",
        "qiskit-terra==0.23.3",
        "pylatexenc",
        "tqdm",
        "uncertainties",
        "pytest",
        "simpy",
        "jupyter"
    ],
    extras_require={
        "docs": [
            "sphinx>=3.0",
            "sphinx-autoapi",
            "numpy>=1.21",
            "pandas>=1.4.0",
        ]
    }
)
