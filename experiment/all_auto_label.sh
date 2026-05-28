#!/bin/bash
# Parallel auto-labeling across (experiment x sensor) combinations.
# Env vars:
#   PARALLEL_JOBS  : how many (experiment, sensor) jobs to run concurrently (default 3)
#   WORKERS_PER_JOB: per-job worker processes (default: cores/PARALLEL_JOBS)
# Expects bg-frame-merged.pcd, bg-frame-os0.pcd, bg-frame-os1.pcd in BASE_DIR.

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
        echo "ERROR: bg frame $bg_file missing"; return 1
    fi
    echo "[$exp/$sensor] starting (workers=$workers)"
    python3 pcd_annotate.py \
        "$pcd_path" \
        --label-dir "$label_dir" \
        --bg-frame "$bg_file" \
        --workers "$workers"
}
export -f run_job
export BASE_DIR

echo "Running with PARALLEL_JOBS=$PARALLEL_JOBS WORKERS_PER_JOB=$WORKERS_PER_JOB (total cores=$TOTAL_CORES)"

JOBS=()
for exp in "${INCLUDE[@]}"; do
    if is_excluded "$exp"; then
        echo "WARN: skipping $exp (excluded)"
        continue
    fi
    JOBS+=("$exp|merged|merged_pcd|merged_labels|bg-frame-merged.pcd")
    JOBS+=("$exp|os0|os0_pcd_transform|os0_labels|bg-frame-os0.pcd")
    JOBS+=("$exp|os1|os1_pcd_transform|os1_labels|bg-frame-os1.pcd")
done

printf '%s\n' "${JOBS[@]}" | \
    xargs -n1 -P"$PARALLEL_JOBS" -I{} bash -c '
        IFS="|" read -r exp sensor pcd_sub lbl_dir bg <<< "$1"
        run_job "$exp" "$sensor" "$pcd_sub" "$lbl_dir" "$bg" "$2"
    ' _ {} "$WORKERS_PER_JOB"

echo "All done."
