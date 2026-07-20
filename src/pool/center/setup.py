from setuptools import find_packages, setup

package_name = 'center'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['detect_interface']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yvonne',
    maintainer_email='yvonne@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	"publisher_dection_boxes=center.publisher_dection_boxes:main",
        	"strategy=center.strategy:main",
        
        
        ],
    },
)
