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
