from setuptools import setup

# Define package metadata
setup(
    name="pthfunctions",
    version="0.0.1",
    description="A library containing plotting and training functionalities",
    author="Yeray Martínez",
    packages=["pthfunctions"],
    package_dir={"pthfunctions": "pthfunctions"},
    url="https://github.com/Yer-Marti/PthFunctions",
    license="MIT",
    install_requires=[
        "torch",
        "tqdm",
        "pandas",
        "matplotlib",
    ],
)