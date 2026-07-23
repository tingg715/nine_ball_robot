from setuptools import find_packages, setup

import os
from glob import glob


package_name = 'yolov7_obj_detect'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        # ROS 2 package index
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        # package.xml
        (
            'share/' + package_name,
            ['package.xml']
        ),

        # Launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]'))
        ),

        # YOLO weights
        (
            os.path.join('share', package_name, 'weights'),
            glob(os.path.join('weights', '*.pt'))
        ),

        # 相機校正檔
        (
            os.path.join('share', package_name, 'config'),
            ['yolov7_obj_detect/camera_calibration.ini']
        ),

        # YOLO model files
        (
            os.path.join('lib', package_name, 'models'),
            [
                'models/experimental.py',
                'models/common.py',
                'models/yolo.py'
            ]
        ),

        # YOLO utility files
        (
            os.path.join('lib', package_name, 'utils'),
            [
                'utils/general.py',
                'utils/torch_utils.py',
                'utils/plots.py',
                'utils/datasets.py',
                'utils/google_utils.py',
                'utils/activations.py',
                'utils/add_nms.py',
                'utils/autoanchor.py',
                'utils/loss.py',
                'utils/metrics.py'
            ]
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='anderson',
    maintainer_email='anderson@todo.todo',

    description='YOLOv7 object detection and camera undistortion node',

    license='TODO: License declaration',

    tests_require=[
        'pytest'
    ],

    entry_points={
        'console_scripts': [
            'object_detection = yolov7_obj_detect.object_detection:main',
            'camera = yolov7_obj_detect.camera:main',
        ],
    },
)