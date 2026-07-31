#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/pointpillars/pointpillars_hv_secfpn_8xb6-50e_exp-crossval-gtsample.py"
DATA_ROOT="${EXP_CV_DATA_ROOT:-data/exp}"
RUN_ROOT="${EXP_CV_RUN_ROOT:-$HOME/runs/pointpillars_crossval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

prepare() {
    "$PYTHON_BIN" tools/dataset_converters/create_exp_crossval_splits.py \
        --root "$DATA_ROOT" --val-ratio 0.1 --guard-frames 10

    for fold in 1 2 3; do
        for view in merged os0 os1; do
            "$PYTHON_BIN" tools/dataset_converters/create_exp_gt_database.py \
                --root "$DATA_ROOT" \
                --info "exp_kitti_infos_${view}_cv${fold}_train.pkl"
        done
    done
}

run_one() {
    local gpu="$1"
    local view="$2"
    local fold="$3"
    local work_dir="$RUN_ROOT/${view}_cv${fold}_gtsample"
    local log="$RUN_ROOT/${view}_cv${fold}_gtsample.log"
    mkdir -p "$work_dir" "results/crossval"

    echo "[$(date --iso-8601=seconds)] START view=$view fold=$fold gpu=$gpu"
    EXP_CV_VIEW="$view" EXP_CV_FOLD="$fold" CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" tools/train.py "$CONFIG" --work-dir "$work_dir" \
        2>&1 | tee "$log"

    local checkpoint
    checkpoint="$(find "$work_dir" -maxdepth 1 \
        -name 'best_osdar23_mAP_epoch_*.pth' -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "$checkpoint" ]]; then
        echo "No best checkpoint found in $work_dir" >&2
        return 1
    fi

    EXP_CV_VIEW="$view" EXP_CV_FOLD="$fold" CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" tools/test.py "$CONFIG" "$checkpoint" \
        --work-dir "$work_dir/test" 2>&1 | tee -a "$log"
    echo "[$(date --iso-8601=seconds)] DONE view=$view fold=$fold checkpoint=$checkpoint"
}

worker_merged() {
    for fold in 1 2 3; do
        run_one 0 merged "$fold"
    done
}

worker_singles() {
    for fold in 1 2 3; do
        run_one 1 os0 "$fold"
        run_one 1 os1 "$fold"
    done
}

launch() {
    mkdir -p "$RUN_ROOT"
    nohup "$0" worker-merged \
        >"$RUN_ROOT/worker_merged.log" 2>&1 &
    echo $! >"$RUN_ROOT/worker_merged.pid"
    nohup "$0" worker-singles \
        >"$RUN_ROOT/worker_singles.log" 2>&1 &
    echo $! >"$RUN_ROOT/worker_singles.pid"
    echo "Launched workers: merged=$(cat "$RUN_ROOT/worker_merged.pid"), singles=$(cat "$RUN_ROOT/worker_singles.pid")"
}

case "${1:-}" in
    prepare) prepare ;;
    worker-merged) worker_merged ;;
    worker-singles) worker_singles ;;
    launch) launch ;;
    *)
        echo "Usage: $0 {prepare|launch|worker-merged|worker-singles}" >&2
        exit 2
        ;;
esac
