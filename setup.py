from setuptools import setup


with open('requirements.txt') as f:
    packages = f.read().splitlines()

setup(
    name='ocrpipeline',
    packages=['ocrpipeline'],
    install_requires=packages
)
