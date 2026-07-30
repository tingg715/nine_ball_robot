# Nine Ball Robot

ROS2 Humble 撞球機器人專案，使用上銀機械手臂、Azure Kinect 與 YOLO 進行球體辨識、座標轉換與擊球控制。

## Environment

- Ubuntu 22.04
- ROS2 Humble
- CUDA 12.4
- Azure Kinect
- Ultralytics YOLO
- Docker

## Build Docker

```bash
cd src/docker
docker build -t nineball_kinect .
./run.sh

export ROS_DOMAIN_ID=1


source ./get_param.sh
docker tag "${DOCKER_HUB_USER}/${IMAGE}:latest" "nineball_kinect:latest"

## 功能

移動到拍照點位

ros2 run hiwin_control move_to_armpos 

移動到白球上方 瞄準方向為子球
ros2 run hiwin_control move_above_cue_aim

平面校正
ros2 run hiwin_control aruco_table_cali_v2 


## Run Nine Ball Script
ros2 launch azure_kinect_ros_driver driver.launch.py 

ros2 run yolov7_obj_detect camera 

ros2 run yolov7_obj_detect object_detection 

ros2 run center publisher_dection_boxes

ros2 run hiwin_libmodbus hiwinlibmodbus_server

ros2 run hiwin_control arm_controller_how 
