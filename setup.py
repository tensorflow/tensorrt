from setuptools import find_packages, setup

setup(
    name='tftrt',
    version='0.0',
    description='NVIDIA TensorRT integration in TensorFlow',
    author='NVIDIA',
    packages=find_packages(),
    install_requires=['tqdm']
)
