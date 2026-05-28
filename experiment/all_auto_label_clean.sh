#!/bin/bash
# Cleaner auto-labeling across (experiment x sensor) combinations.
# Writes new label folders and does not overwrite the original labels.
#
# Main differences to all_auto_label.sh:
# - no hard-coded static labels
# - only the expected dynamic class per experiment is kept
# - dynamic box dimensions are stabilized via class templates
# - only the largest plausible dynamic cluster per frame is kept
#
# Env vars:
#   PARALLEL_JOBS  : how many (experiment, sensor) jobs to run concurrently (default 3)
#   WORKERS_PER_JOB: per-job worker processes (default: cores/PARALLEL_JOBS)

set -euo pipefail
BASE_DIR=""
INCLUDE=()
EXCLUDE=()
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
TOTAL_CORES="$(nproc)"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-$(( TOTAL_CORES / PARALLEL_JOBS ))}"
[[ "$WORKERS_PER_JOB" -lt 1 ]] && WORKERS_PER_JOB=1

usage() {
    echo "Usage: $0 --base /path/to/base [--include a,b] [--exclude c,d]"
    echo "Env: PARALLEL_JOBS=$PARALLEL_JOBS WORKERS_PER_JOB=$WORKERS_PER_JOB"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --base) shift; BASE_DIR="$1" ;;
        --include) shift; IFS=',' read -r -a INCLUDE <<< "$1" ;;
        --exclude) shift; IFS=',' read -r -a EXCLUDE <<< "$1" ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

[[ -z "$BASE_DIR" ]] && { echo "ERROR: --base required."; usage; }
[[ ! -d "$BASE_DIR" ]] && { echo "ERROR: base directory '$BASE_DIR' does not exist."; exit 1; }

ALL_DIRS=(
    "1_experiment_car_1" "2_experiment_car_2" "3_experiment_car_3"
    "4_experiment_bike_1" "5_experiment_bike_3" "6_experiment_bike_4"
    "7_experiment_person_1" "8_experiment_person_2" "9_experiment_person_3"
)

[[ ${#INCLUDE[@]} -eq 0 ]] && INCLUDE=("${ALL_DIRS[@]}")

is_excluded() {
    local d=$1
    for ex in "${EXCLUDE[@]:-}"; do [[ "$ex" == "$d" ]] && return 0; done
    return 1
}

expected_class_for_exp() {
    local exp="$1"
    if [[ "$exp" == *"_car_"* ]]; then
        echo "car"
    elif [[ "$exp" == *"_bike_"* ]]; then
        echo "bike"
    elif [[ "$exp" == *"_person_"* ]]; then
        echo "person"
    else
        echo "ERROR"
        return 1
    fi
}

min_points_for_class() {
    local cls="$1"
    case "$cls" in
        car) echo 160 ;;
        bike) echo 80 ;;
        person) echo 45 ;;
        *) echo 120 ;;
    esac
}

eps_for_class() {
    local cls="$1"
    case "$cls" in
        car) echo 0.99 ;;
        bike) echo 0.75 ;;
        person) echo 0.65 ;;
        *) echo 0.99 ;;
    esac
}

run_job() {
    local exp="$1"
    local sensor="$2"
    local pcd_subdir="$3"
    local label_dir="$4"
    local bg_file="$5"
    local workers="$6"
    cd "$BASE_DIR"

    local pcd_path="$exp/$pcd_subdir"
    if [[ ! -d "$pcd_path" ]]; then
        echo "WARN: $pcd_path missing, skipping"
        return 0
    fi
    if [[ ! -f "$bg_file" ]]; then
        echo "ERROR: bg frame $bg_file missing"
        return 1
    fi

    local expected
    expected="$(expected_class_for_exp "$exp")"
    local min_points
    min_points="$(min_points_for_class "$expected")"
    local eps
    eps="$(eps_for_class "$expected")"

    echo "[$exp/$sensor] starting clean labels (class=$expected workers=$workers)"
    python3 pcd_annotate.py \
        "$pcd_path" \
        --label-dir "$label_dir" \
        --bg-frame "$bg_file" \
        --workers "$workers" \
        --no-static-labels \
        --expected-class "$expected" \
        --use-template-dims \
        --keep-largest 1 \
        --bg-threshold 0.05 \
        --max-foreground-ratio 15 \
        --dbscan-eps "$eps" \
        --min-cluster-points "$min_points" \
        --exclude-origin-radius 2.0
}
export -f run_job expected_class_for_exp min_points_for_class eps_for_class
export BASE_DIR

echo "Running clean labels with PARALLEL_JOBS=$PARALLEL_JOBS WORKERS_PER_JOB=$WORKERS_PER_JOB (total cores=$TOTAL_CORES)"

JOBS=()
for exp in "${INCLUDE[@]}"; do
    if is_excluded "$exp"; then
        echo "WARN: skipping $exp (excluded)"
        continue
    fi
    JOBS+=("$exp|merged|merged_pcd|merged_labels_clean|bg-frame-merged.pcd")
    JOBS+=("$exp|os0|os0_pcd_transform|os0_labels_clean|bg-frame-os0.pcd")
    JOBS+=("$exp|os1|os1_pcd_transform|os1_labels_clean|bg-frame-os1.pcd")
done

printf '%s\n' "${JOBS[@]}" | \
    xargs -n1 -P"$PARALLEL_JOBS" -I{} bash -c '
        IFS="|" read -r exp sensor pcd_sub lbl_dir bg <<< "$1"
        run_job "$exp" "$sensor" "$pcd_sub" "$lbl_dir" "$bg" "$2"
    ' _ {} "$WORKERS_PER_JOB"

echo "All clean labeling jobs done."
