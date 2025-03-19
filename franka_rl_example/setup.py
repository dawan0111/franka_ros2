import os
import glob
from setuptools import find_packages, setup

package_name = 'franka_rl_example'
model_files = glob.glob('models/**/*', recursive=True)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', model_files),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'onnxruntime'
    ],
    zip_safe=True,
    maintainer='airo',
    maintainer_email='msd030428@gmail.com',
    description='A simple ROS2 package for controlling Franka arm robot using reinforcement learning',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'reach_franka_example_node = franka_rl_example.reach_franka_example_node:main',
        ],
    },
)
