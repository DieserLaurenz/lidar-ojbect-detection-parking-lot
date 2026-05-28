#!/bin/bash
# Parallel point cloud merge across experiments.
# Env vars:
#   PARALLEL_EXPS  : how many experiments to run concurrently (default 3)
#   WORKERS_PER_EXP: per-experiment worker threads (default: cores/PARALLEL_EXPS)

set -euo pipefail
BASE_DIR=""
INCLUDE=()
EXCLUDE=()
PARALLEL_EXPS="${PARALLEL_EXPS:-3}"
TOTAL_CORES="$(nproc)"
WORKERS_PER_EXP="${WORKERS_PER_EXP:-$(( TOTAL_CORES / PARALLEL_EXPS ))}"
[[ "$WORKERS_PER_EXP" -lt 1 ]] && WORKERS_PER_EXP=1

usage() {
    echo "Usage: $0 --base /path/to/base [--include a,b] [--exclude c,d]"
    echo "Env: PARALLEL_EXPS=$PARALLEL_EXPS WORKERS_PER_EXP=$WORKERS_PER_EXP"
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

process_one() {
    local exp="$1"
    local workers="$2"
    cd "$BASE_DIR"
    if [[ ! -d "$exp" ]]; then
        echo "WARN: $exp not found, skipping"
        return 0
    fi
    if [[ ! -d "$exp/os0_pcd" || ! -d "$exp/os1_pcd" ]]; then
        echo "WARN: $exp missing os0_pcd/os1_pcd, skipping"
        return 0
    fi
    echo "[$exp] starting (workers=$workers)"
    python3 pcd_merge.py \
        "$exp/os0_pcd" \
        "$exp/os1_pcd" \
        --merged-dir "merged_pcd" \
        --csv-path "$exp/merge_times.csv" \
        --max-worker "$workers"
}
export -f process_one
export BASE_DIR

echo "Running with PARALLEL_EXPS=$PARALLEL_EXPS WORKERS_PER_EXP=$WORKERS_PER_EXP (total cores=$TOTAL_CORES)"

TO_RUN=()
for exp in "${INCLUDE[@]}"; do
    if is_excluded "$exp"; then
        echo "WARN: skipping $exp (excluded)"
        continue
    fi
    TO_RUN+=("$exp")
done

printf '%s\n' "${TO_RUN[@]}" | \
    xargs -n1 -P"$PARALLEL_EXPS" -I{} bash -c 'process_one "$@"' _ {} "$WORKERS_PER_EXP"

echo "All done."
