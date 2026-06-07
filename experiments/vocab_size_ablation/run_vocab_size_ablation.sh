#!/bin/bash
#
# Vocab-size ablation: trains EZPC for every (seed, num_concepts) combination,
# then evaluates each vocab size across all seeds with test_all.py.
#
# Usage:
#   bash run_vocab_size_ablation.sh \
#     --dataset CIFAR-100 \
#     --dataset_root /path/to/data \
#     --seeds "12,123,1234" \
#     --vocab_sizes "250,500,1000,2000,3000"
#
# Optional overrides (shown with defaults):
#     --backbone    RN50
#     --num_epochs  10000
#     --lr          0.01
#     --lambda_weight 1
#     --batch_size  512
#     --device      cuda
#     --output_path ./outputs

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
DATASET=""
DATASET_ROOT=""
SEEDS="12,123,1234"
VOCAB_SIZES="250,500,1000,2000,3000"
BACKBONE="RN50"
NUM_EPOCHS=10000
LR=0.01
LAMBDA_WEIGHT=1
BATCH_SIZE=512
DEVICE="cuda"
OUTPUT_PATH="./outputs"

# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="$2";        shift 2 ;;
        --dataset_root)   DATASET_ROOT="$2";   shift 2 ;;
        --seeds)          SEEDS="$2";          shift 2 ;;
        --vocab_sizes)    VOCAB_SIZES="$2";    shift 2 ;;
        --backbone)       BACKBONE="$2";       shift 2 ;;
        --num_epochs)     NUM_EPOCHS="$2";     shift 2 ;;
        --lr)             LR="$2";             shift 2 ;;
        --lambda_weight)  LAMBDA_WEIGHT="$2";  shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --device)         DEVICE="$2";         shift 2 ;;
        --output_path)    OUTPUT_PATH="$2";    shift 2 ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

# ── Validate required arguments ───────────────────────────────────
if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    exit 1
fi
if [[ -z "$DATASET_ROOT" ]]; then
    echo "Error: --dataset_root is required"
    exit 1
fi

# ── Resolve script directory (so we can call train.py / test_all.py) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Convert comma-separated lists to arrays ───────────────────────
IFS=',' read -ra SEED_ARR  <<< "$SEEDS"
IFS=',' read -ra M_ARR     <<< "$VOCAB_SIZES"

echo "============================================="
echo "  Vocab-Size Ablation"
echo "============================================="
echo "  Dataset:      $DATASET"
echo "  Dataset root: $DATASET_ROOT"
echo "  Backbone:     $BACKBONE"
echo "  Seeds:        ${SEED_ARR[*]}"
echo "  Vocab sizes:  ${M_ARR[*]}"
echo "  Epochs:       $NUM_EPOCHS"
echo "  LR:           $LR"
echo "  Lambda:       $LAMBDA_WEIGHT"
echo "  Batch size:   $BATCH_SIZE"
echo "  Device:       $DEVICE"
echo "  Output path:  $OUTPUT_PATH"
echo "============================================="

# ── Phase 1: Training ────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Phase 1: Training"
echo "============================================="

for num_concepts in "${M_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
        # Strip whitespace
        num_concepts="${num_concepts// /}"
        seed="${seed// /}"

        echo ""
        echo ">>> Training num_concepts=${num_concepts}, seed=${seed}"
        echo "---------------------------------------------"

        python "${SCRIPT_DIR}/train.py" \
            --dataset        "$DATASET" \
            --dataset_root   "$DATASET_ROOT" \
            --backbone       "$BACKBONE" \
            --num_epochs     "$NUM_EPOCHS" \
            --lr             "$LR" \
            --lambda_weight  "$LAMBDA_WEIGHT" \
            --batch_size     "$BATCH_SIZE" \
            --num_concepts   "$num_concepts" \
            --subset_seed    "$seed" \
            --output_path    "$OUTPUT_PATH" \
            --device         "$DEVICE"

        echo ">>> Finished training num_concepts=${num_concepts}, seed=${seed}"
    done
done

echo ""
echo "============================================="
echo "  Phase 1 complete - all models trained."
echo "============================================="

# ── Phase 2: Evaluation ──────────────────────────────────────────
echo ""
echo "============================================="
echo "  Phase 2: Evaluation (test_all.py)"
echo "============================================="

for num_concepts in "${M_ARR[@]}"; do
    num_concepts="${num_concepts// /}"

    echo ""
    echo ">>> Evaluating num_concepts=${num_concepts} across seeds: ${SEED_ARR[*]}"
    echo "---------------------------------------------"

    python "${SCRIPT_DIR}/test_all.py" \
        --dataset          "$DATASET" \
        --dataset_root     "$DATASET_ROOT" \
        --backbone         "$BACKBONE" \
        --num_epochs       "$NUM_EPOCHS" \
        --lr               "$LR" \
        --lambda_weight    "$LAMBDA_WEIGHT" \
        --train_batch_size "$BATCH_SIZE" \
        --eval_batch_size  "$BATCH_SIZE" \
        --num_concepts     "$num_concepts" \
        --seeds            "$SEEDS" \
        --output_path      "$OUTPUT_PATH" \
        --device           "$DEVICE"

    echo ">>> Finished evaluating num_concepts=${num_concepts}"
done

echo ""
echo "============================================="
echo "  Phase 2 complete - all evaluations done."
echo "============================================="
echo ""
echo "All results saved under: $OUTPUT_PATH"
