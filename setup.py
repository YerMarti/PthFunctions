from setuptools import setup, find_packages

# Define package metadata
setup(
    name="pthfunctions",
    version="0.0.1",
    description="A library containing plotting and training functionalities",
    author="Yeray Martínez",
    packages=find_packages(where="PthFunctions"),
    include_package_data=True,
    install_requires=[
        "torch",
        "tqdm.auto",
        "pandas",
        "matplotlib.pyplot",
    ],
)