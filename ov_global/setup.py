from setuptools import find_packages, setup

package_name = 'ov_global'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ov_global.launch.py']),
        ('share/' + package_name + '/config', ['config/zed_rear_left.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eivis',
    maintainer_email='eivis@users.noreply.github.com',
    description='Global alignment and AprilTag pose fusion nodes for OpenVINS.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tag_pnp_node = ov_global.tag_pnp_node:main',
            'global_alignment_node = ov_global.global_alignment_node:main',
        ],
    },
)
