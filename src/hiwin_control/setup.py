from setuptools import find_packages, setup

package_name = 'hiwin_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # 校正檔（.ini/.yaml）跟 .py 放在同一個資料夾，程式用 __file__ 推路徑找它們，
    # 所以安裝時必須一起複製到 site-packages/hiwin_control/，否則執行期會找不到。
    package_data={package_name: ['*.ini', '*.yaml']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zack',
    maintainer_email='zack@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_controller = hiwin_control.arm_controller:main',
            'arm_controller_how = hiwin_control.arm_controler_how:main',
            'arm_controller_track = hiwin_control.arm_controler_track:main',
            'arm_controller_v2 = hiwin_control.arm_controller_v2:main',
            'team_selection = hiwin_control.team_selection:main',
            'testbeat_arm_controller = hiwin_control.testbeat_arm_controller:main',
            'pool_arm_controller = hiwin_control.pool_arm_controller:main',
            'nine_arm_controller = hiwin_control.nine_arm_controller:main',
            'rush_arm_controller = hiwin_control.rush_arm_controller:main',
            'yyy_arm_controller = hiwin_control.yyy_arm_controller:main',
            'stream_rs = hiwin_control.stream_rs:main',
            'aruco_table_cali_v2 = hiwin_control.aruco_table_cali_v2:main',
            'table_cali_from_balls = hiwin_control.table_cali_from_balls:main',
            'my_pool = hiwin_control.my_pool_1:main',
            'billiards_planner = hiwin_control.billiards_planner_ui:main',
            'test_camera_mm = hiwin_control.test_camera_mm:main',
            'ball_coordinate_checker = hiwin_control.ball_coordinate_checker:main',
            'ball_position_test = hiwin_control.ball_position_test:main',
            'move_to_armpos = hiwin_control.move_to_armpos:main',
            'arm_controller_ball_coordinate = hiwin_control.arm_controller_ball_coordinate:main',
            'move_above_cue_aim = hiwin_control.move_above_cue_aim:main',

        ],
    },
)
