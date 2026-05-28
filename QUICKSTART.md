# Quickstart: Multi-Perspective LiDAR Visualization (WSL2 / ROS2)

This guide describes how to visualize the LiDAR point cloud data using ROS2 inside WSL2.

## 1. Prerequisites
* **OS:** Windows with WSL2 (Ubuntu 24.04).
* **Software:** ROS2 (Jazzy) installed.
* **Path Note:** In WSL, your C: drive is located at `/mnt/c/`.

## 2. Setting the Project Path
Open your Ubuntu terminal and set a variable for your project directory (makes navigation easier):
```bash
# Copy and paste this line
export PROJ_DIR="/mnt/c/Users/laure/Documents/Uni Sachen/TU Berlin/4. Semester/DCAITI Projekt/Multisensor LiDAR Object Detection/Masterarbeit"
cd "$PROJ_DIR"
```

## 3. Preparation
Source the ROS2 environment in every new terminal:
```bash
source /opt/ros/jazzy/setup.bash
```

Extract an experiment (if not already done):
```bash
cd "$PROJ_DIR"
mkdir -p raw_data
tar -xf "data/2025_10_09/Experiment-Data/experiments/raw_1_experiment_car_1.tar.xz" -C raw_data/
```

## 4. Playback (Terminal 1)
Navigate into the extracted bag folder and start playback:
```bash
cd "$PROJ_DIR/raw_data/1_experiment_car_1/os0_rosbag2_2025_10_09-09_29_02"
ros2 bag play . -l
```

## 5. Visualization (Terminal 2)
In a new Ubuntu terminal:
```bash
source /opt/ros/jazzy/setup.bash
export PROJ_DIR="/mnt/c/Users/laure/Documents/Uni Sachen/TU Berlin/4. Semester/DCAITI Projekt/Multisensor LiDAR Object Detection/Masterarbeit"
rviz2 -d "$PROJ_DIR/ouster_view.rviz"
```

**Required RViz Settings (if not using the .rviz file):**
1. **Fixed Frame:** `os_lidar`
2. **Add Display:** *Add* -> *By Topic* -> `/ouster_os0/points`
3. **QoS:** Reliability: `Reliable`, Durability: `Transient Local`
4. **View:** *Color Transformer:* `AxisColor`, *Size:* `0.02`

## 6. Pro-Tip: Permanent Path
Add the project path to your `.bashrc` so you don't have to export it every time:
```bash
echo 'export PROJ_DIR="/mnt/c/Users/laure/Documents/Uni Sachen/TU Berlin/4. Semester/DCAITI Projekt/Multisensor LiDAR Object Detection/Masterarbeit"' >> ~/.bashrc
source ~/.bashrc
```
Then you can always just type `cd "$PROJ_DIR"`.
