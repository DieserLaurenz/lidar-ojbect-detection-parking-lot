#!/usr/bin/env bash
# CenterPoint-Pendant zu run_exp_crossval.sh: gleiche Folds/Splits/GT-Datenbanken
# (bereits durch "run_exp_crossval.sh prepare" erzeugt), nur andere Architektur.
set -euo pipefail

CONFIG="configs/centerpoint/centerpoint_pillar016_second_secfpn_8xb6-50e_exp-crossval-gtsample.py"
RUN_ROOT="${EXP_CV_RUN_ROOT:-$HOME/runs/centerpoint_crossval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_one() {
    local gpu="$1"
    local view="$2"
    local fold="$3"
    local work_dir="$RUN_ROOT/${view}_cv${fold}_gtsample"
    local log="$RUN_ROOT/${view}_cv${fold}_gtsample.log"
    mkdir -p "$work_dir" "results/crossval_cp"

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

case "${1:-}" in
    worker-merged) worker_merged ;;
    worker-singles) worker_singles ;;
    *)
        echo "Usage: $0 {worker-merged|worker-singles}" >&2
        exit 2
        ;;
esac
