# Nine Ball Robot

ROS2 Humble 撞球機器人專案，使用上銀機械手臂、Azure Kinect 與 YOLO 進行球體辨識、座標轉換與擊球控制。

## 環境需求 (Environment)

- Ubuntu 22.04
- ROS2 Humble
- CUDA 12.4
- Azure Kinect SDK
- Ultralytics YOLO
- Docker

## 安裝與建置 (Build)

### 1. Build Docker Image

```bash
cd src/docker
docker build -t nineball_kinect .
./run.sh
```

### 2. 環境變數設定

```bash
export ROS_DOMAIN_ID=1
source ./get_param.sh
docker tag "${DOCKER_HUB_USER}/${IMAGE}:latest" "nineball_kinect:latest"
```

## 功能節點 (Functions)

各節點可依需求單獨執行：

**移動到拍照點位**
```bash
ros2 run hiwin_control move_to_armpos
```

**移動到白球上方，瞄準方向為子球**
```bash
ros2 run hiwin_control move_above_cue_aim
```

**平面校正**
```bash
ros2 run hiwin_control aruco_table_cali_v2
```

## 完整啟動流程 (Run Nine Ball Script)

依序在各個終端機執行以下指令（建議依序啟動，確認前一個節點正常運作後再開下一個）：

1. 啟動 Azure Kinect 驅動
   ```bash
   ros2 launch azure_kinect_ros_driver driver.launch.py
   ```

2. 啟動 realsense 相機
   ```bash
   ros2 launch hiwin_control wrist_cam.launch.py
   ```

3. 啟動相機節點
   ```bash
   ros2 run yolov7_obj_detect camera
   ```

4. 啟動物件偵測
   ```bash
   ros2 run yolov7_obj_detect object_detection
   ```

5. 啟動偵測框發布節點
   ```bash
   ros2 run center publisher_dection_boxes
   ```

6. 啟動 Modbus 通訊伺服器
   ```bash
   ros2 run hiwin_libmodbus hiwinlibmodbus_server
   ```

7. 啟動手臂控制器
   ```bash
   ros2 run hiwin_control arm_controller_how
   ```


## 專案結構 (Project Structure)

> TODO：補上主要資料夾說明，例如：
> ```
> src/
> ├── docker/           # Docker 建置檔案
> ├── hiwin_control/     # 機械手臂控制相關套件
> ├── yolov7_obj_detect/ # 球體偵測
> └── ...
> ```

## Team

> TODO：補上團隊成員與分工

## License

> TODO：補上授權說明（若適用）
