from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name = "Smart Manufacturing Machine Efficiency Predictor",
    version = "0.1.0",
    author = "Amit Singh",
    author_email = "amitksingh3022@gmail.com",
    packages = find_packages(),
    install_requires = requirements,
)
