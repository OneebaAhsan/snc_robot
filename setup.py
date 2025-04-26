import glob
import os
from setuptools import find_packages, setup

package_name = 'snc_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Install marker template images
        (os.path.join('share', package_name, 'markers/hazard_template'), glob.glob(os.path.join('markers/hazard_template', '*.png'))),
        # Install marker session file
        (os.path.join('share', package_name, 'find_object_session'), glob.glob(os.path.join('find_object_session', '*.bin'))),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node2 = snc_robot.node2:main',  # This should correctly point to the main function in node2.py
        ],
    },
)


