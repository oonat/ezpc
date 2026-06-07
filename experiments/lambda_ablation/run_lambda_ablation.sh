#!/bin/bash
#
# Lambda-weight ablation: trains EZPC for every lambda value,
# then evaluates each model with test.py and prints a summary table.
#
# Usage:
#   bash run_lambda_ablation.sh \
#     --dataset CIFAR-100 \
#     --dataset_root /path/to/data \
#     --lambda_values "0.01,0.1,1,10,100,1000"
#
# Optional overrides (shown with defaults):
#     --backbone       RN50
#     --num_epochs     10000
#     --lr             0.01
#     --batch_size     1000000
#     --eval_batch_size 512
#     --device         cuda
#     --output_path    ./lambda_ablation_results

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
DATASET=""
DATASET_ROOT=""
LAMBDA_VALUES="0.01,0.1,1,10,100,1000"
BACKBONE="RN50"
NUM_EPOCHS=10000
LR=0.01
BATCH_SIZE=1000000
EVAL_BATCH_SIZE=512
DEVICE="cuda"
OUTPUT_PATH="./lambda_ablation_results"

# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)          DATASET="$2";          shift 2 ;;
        --dataset_root)     DATASET_ROOT="$2";     shift 2 ;;
        --lambda_values)    LAMBDA_VALUES="$2";    shift 2 ;;
        --backbone)         BACKBONE="$2";         shift 2 ;;
        --num_epochs)       NUM_EPOCHS="$2";       shift 2 ;;
        --lr)               LR="$2";               shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";       shift 2 ;;
        --eval_batch_size)  EVAL_BATCH_SIZE="$2";  shift 2 ;;
        --device)           DEVICE="$2";           shift 2 ;;
        --output_path)      OUTPUT_PATH="$2";      shift 2 ;;
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

# ── Resolve paths ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Convert backbone to filesystem-safe name (mirrors utils.backbone_to_name) ──
if echo "$BACKBONE" | grep -qi "siglip"; then
    BACKBONE_NAME="siglip-so400m-patch14-384"
else
    BACKBONE_NAME="${BACKBONE//\//-}"
fi

# ── Convert comma-separated list to array ────────────────────────
IFS=',' read -ra LAMBDA_ARR <<< "$LAMBDA_VALUES"

# ── Helper: convert a lambda value to the string Python's float() produces ──
# This ensures checkpoint path matching (e.g., "1" -> "1.0", "0.01" -> "0.01")
to_py_float() {
    python3 -c "print(float($1))"
}

echo "============================================="
echo "  Lambda-Weight Ablation"
echo "============================================="
echo "  Dataset:        $DATASET"
echo "  Dataset root:   $DATASET_ROOT"
echo "  Backbone:       $BACKBONE ($BACKBONE_NAME)"
echo "  Lambda values:  ${LAMBDA_ARR[*]}"
echo "  Epochs:         $NUM_EPOCHS"
echo "  LR:             $LR"
echo "  Train BS:       $BATCH_SIZE"
echo "  Eval BS:        $EVAL_BATCH_SIZE"
echo "  Device:         $DEVICE"
echo "  Output path:    $OUTPUT_PATH"
echo "============================================="

# ── Phase 1: Training ────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Phase 1: Training"
echo "============================================="

for lam in "${LAMBDA_ARR[@]}"; do
    lam="${lam// /}"

    echo ""
    echo ">>> Training lambda=${lam}"
    echo "---------------------------------------------"

    python "${ROOT_DIR}/train.py" \
        --dataset        "$DATASET" \
        --dataset_root   "$DATASET_ROOT" \
        --backbone       "$BACKBONE" \
        --num_epochs     "$NUM_EPOCHS" \
        --lr             "$LR" \
        --lambda_weight  "$lam" \
        --batch_size     "$BATCH_SIZE" \
        --device         "$DEVICE"

    echo ">>> Finished training lambda=${lam}"
done

echo ""
echo "============================================="
echo "  Phase 1 complete - all models trained."
echo "============================================="

# ── Phase 2: Evaluation ──────────────────────────────────────────
echo ""
echo "============================================="
echo "  Phase 2: Evaluation"
echo "============================================="

mkdir -p "$OUTPUT_PATH"

for lam in "${LAMBDA_ARR[@]}"; do
    lam="${lam// /}"
    lam_py="$(to_py_float "$lam")"

    # Construct checkpoint path to match train.py output format
    # train.py: f"checkpoints/{dataset}_backbone_{name}_weight_{lambda}_epoch_{epochs}_lr_{lr}_bs_{bs}"
    CKPT_DIR="checkpoints/${DATASET}_backbone_${BACKBONE_NAME}_weight_${lam_py}_epoch_${NUM_EPOCHS}_lr_${LR}_bs_${BATCH_SIZE}"
    CKPT_PATH="${CKPT_DIR}/best_A.pth"

    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "WARNING: Checkpoint not found: $CKPT_PATH - skipping lambda=${lam}"
        continue
    fi

    # Use a per-lambda output subdir to avoid overwriting results
    LAM_OUTPUT="${OUTPUT_PATH}/lambda_${lam_py}"

    echo ""
    echo ">>> Evaluating lambda=${lam} (checkpoint: ${CKPT_DIR})"
    echo "---------------------------------------------"

    python "${ROOT_DIR}/test.py" \
        --dataset          "$DATASET" \
        --dataset_root     "$DATASET_ROOT" \
        --backbone         "$BACKBONE" \
        --batch_size       "$EVAL_BATCH_SIZE" \
        --checkpoint_path  "$CKPT_PATH" \
        --output_path      "$LAM_OUTPUT" \
        --device           "$DEVICE"

    echo ">>> Finished evaluating lambda=${lam}"
done

echo ""
echo "============================================="
echo "  Phase 2 complete - all evaluations done."
echo "============================================="

# ── Phase 3: Summary Table ───────────────────────────────────────
echo ""
echo "============================================="
echo "  Lambda Ablation Results: ${DATASET} / ${BACKBONE}"
echo "============================================="

# Helper: extract a metric value from a result file
extract_metric() {
    local file="$1"
    local key="$2"
    grep "^${key}:" "$file" | head -1 | awk -F': ' '{print $2}'
}

# Print table header
printf "%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-12s | %-8s | %-8s\n" \
    "Lambda" "Seen" "Unseen" "G-Seen" "G-Unseen" "H" "Top1Agree(%)" "Spearman" "KL"
printf "%-10s-+-%-8s-+-%-8s-+-%-8s-+-%-8s-+-%-8s-+-%-12s-+-%-8s-+-%-8s\n" \
    "----------" "--------" "--------" "--------" "--------" "--------" "------------" "--------" "--------"

# Also save the table to a file
TABLE_FILE="${OUTPUT_PATH}/summary_table.txt"
{
    printf "%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-12s | %-8s | %-8s\n" \
        "Lambda" "Seen" "Unseen" "G-Seen" "G-Unseen" "H" "Top1Agree(%)" "Spearman" "KL"
    printf "%-10s-+-%-8s-+-%-8s-+-%-8s-+-%-8s-+-%-8s-+-%-12s-+-%-8s-+-%-8s\n" \
        "----------" "--------" "--------" "--------" "--------" "--------" "------------" "--------" "--------"
} > "$TABLE_FILE"

for lam in "${LAMBDA_ARR[@]}"; do
    lam="${lam// /}"
    lam_py="$(to_py_float "$lam")"

    RESULT_FILE="${OUTPUT_PATH}/lambda_${lam_py}/${DATASET}_${BACKBONE_NAME}_results.txt"

    if [[ ! -f "$RESULT_FILE" ]]; then
        printf "%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-12s | %-8s | %-8s\n" \
            "$lam" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A"
        continue
    fi

    SEEN=$(extract_metric "$RESULT_FILE" "EZPC Seen Accuracy")
    UNSEEN=$(extract_metric "$RESULT_FILE" "EZPC Unseen Accuracy")
    G_SEEN=$(extract_metric "$RESULT_FILE" "EZPC Generalized Seen Accuracy")
    G_UNSEEN=$(extract_metric "$RESULT_FILE" "EZPC Generalized Unseen Accuracy")
    H=$(extract_metric "$RESULT_FILE" "EZPC Generalized Harmonic Mean")
    TOP1=$(extract_metric "$RESULT_FILE" "Fidelity All Top1Agree (%)")
    SPEARMAN=$(extract_metric "$RESULT_FILE" "Fidelity All Spearman")
    KL=$(extract_metric "$RESULT_FILE" "Fidelity All KL")

    # Print to stdout
    printf "%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-12s | %-8s | %-8s\n" \
        "$lam" "$SEEN" "$UNSEEN" "$G_SEEN" "$G_UNSEEN" "$H" "$TOP1" "$SPEARMAN" "$KL"

    # Append to file
    printf "%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-12s | %-8s | %-8s\n" \
        "$lam" "$SEEN" "$UNSEEN" "$G_SEEN" "$G_UNSEEN" "$H" "$TOP1" "$SPEARMAN" "$KL" >> "$TABLE_FILE"
done

echo "============================================="
echo ""
echo "Summary table saved to: ${TABLE_FILE}"
echo "Per-lambda results saved under: ${OUTPUT_PATH}/"
