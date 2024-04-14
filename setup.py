from setuptools import setup, find_packages

# Define package metadata
setup(
    name="pthfunctions",
    version="0.0.1",
    description="A library containing plotting and training functionalities",
    author="Yeray Martínez",
    packages=find_packages(where="PthFunctions"),
    url="https://github.com/Yer-Marti/PthFunctions",
    license="MIT",
    install_requires=[
        "torch",
        "tqdm",
        "pandas",
        "matplotlib",
    ],
)