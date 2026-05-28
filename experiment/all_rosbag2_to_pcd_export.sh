#!/bin/bash

set -euo pipefail
BASE_DIR=""
ALL_DIRS=()
INCLUDE=()
EXCLUDE=()
function usage() {
    echo "Usage: $0 --base /path/to/base [--include subdir1,subdir2] [--exclude subdir3,subdir4]"
    echo
    echo "Example:"
    echo "  $0 --base ./pointclouds --exclude subdir1,subdir2"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --base)
            shift
            BASE_DIR="$1"
            ;;
        --include)
            shift
            IFS=',' read -r -a INCLUDE <<< "$1"
            ;;
        --exclude)
            shift
            IFS=',' read -r -a EXCLUDE <<< "$1"
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
    shift
done

if [[ -z "$BASE_DIR" ]]; then
    echo "ERROR: --base argument is required."
    usage
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: base directory '$BASE_DIR' does not exist."
    exit 1
fi

ALL_DIRS=(
    "1_experiment_car_1"
    "2_experiment_car_2"
    "3_experiment_car_3"
    "4_experiment_bike_1"
    "5_experiment_bike_3"
    "6_experiment_bike_4"
    "7_experiment_person_1"
    "8_experiment_person_2"
    "9_experiment_person_3" 
)

if [[ ${#INCLUDE[@]} -eq 0 ]]; then
    INCLUDE=("${ALL_DIRS[@]}")
fi

function is_excluded() {
    local dir=$1
    for ex in "${EXCLUDE[@]}"; do
        [[ "$ex" == "$dir" ]] && return 0
    done
    return 1
}

for base_dir in "${INCLUDE[@]}"; do
    if is_excluded "$base_dir"; then
        echo "WARN: Skipping $dir (excluded)"
        continue
    fi
    
    rosbag_dirs=($(find "$base_dir" -maxdepth 1 -type d -name "*rosbag2*" -printf "%f\n" | sort))
    count=${#rosbag_dirs[@]}

    if [ ! "$count" -eq 2 ]; then
        echo "Error: Expected to find two rosbag2 in $base_dir, but found $cound"
        exit 1
    fi
    
    python3 ros2_exporter.py "${rosbag_dirs[0]}" "${rosbag_dirs[1]}"
