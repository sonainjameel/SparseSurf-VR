#!/bin/bash
#PBS -N ss_small21_p7v1_10k
#PBS -q large_gpuq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=05:00:00
#PBS -j oe

set -eo pipefail

REPO="$HOME/projects/SparseSurf-VR"

SCENE="$HOME/sparsesurf_shared/data/mipnerf360/SmallRoom_eval24"

CKPT="$HOME/sparsesurf_shared/checkpoints/23-51-11/model_best_bp2.pth"

MODEL="$HOME/sparsesurf_shared/outputs/smallroom21_stable_post7k_v1_10000_fullres"

echo "============================================================"
echo "SparseSurf — SmallRoom — 21 views — stable post-7k V1 — 10k"
echo "============================================================"
echo "Job:          $PBS_JOBID"
echo "Node:         $(hostname)"
echo "Scene:        $SCENE"
echo "Output:       $MODEL"
echo "Train views:  21"
echo "Test views:   3 fixed"
echo "Iterations:   10000"
echo "Resolution:   1"
echo "Virtual cams: 240"
echo "============================================================"

# ------------------------------------------------------------
# Do not overwrite an existing experiment
# ------------------------------------------------------------

if [ -e "$MODEL" ]; then
    echo "ERROR: Output already exists:"
    echo "$MODEL"
    exit 1
fi

# ------------------------------------------------------------
# Required input checks
# ------------------------------------------------------------

for REQUIRED in \
    "$SCENE/poses_bounds.npy" \
    "$SCENE/sparse/0/cameras.bin" \
    "$SCENE/sparse/0/images.bin" \
    "$SCENE/sparse/0/points3D.bin" \
    "$SCENE/21_views/dense/fused.ply" \
    "$SCENE/21_views/pair.txt" \
    "$SCENE/splits/train_21.txt" \
    "$SCENE/splits/test_fixed.txt" \
    "$CKPT"
do
    if [ ! -e "$REQUIRED" ]; then
        echo "ERROR: Missing:"
        echo "$REQUIRED"
        exit 1
    fi
done

echo
echo "Dataset checks:"
echo -n "Images: "
find "$SCENE/images" -maxdepth 1 -name "*.png" | wc -l

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source "$REPO/activate_sparsesurf_bw.sh"
export PYTHONPATH="$REPO/submodules/diff-plane-rasterization:${PYTHONPATH:-}"

cd "$REPO"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

echo
nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible CUDA devices:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY

# ============================================================
# STAGE 1 — TRAIN 7,000 ITERATIONS
# ============================================================

echo
echo "============================================================"
echo "STAGE 1: SPARSESURF 10K TRAINING"
echo "============================================================"

python -u train.py \
    -s "$SCENE" \
    -m "$MODEL" \
    -r 1 \
    --eval \
    --n_views 21 \
    --iterations 10000 \
    --test_iterations 10000 \
    --save_iterations 10000 \
    --total_virtual_num 240 \
    --foundation_stereo_ckpt "$CKPT"

POINT_CLOUD="$MODEL/point_cloud/iteration_10000/point_cloud.ply"

if [ ! -f "$POINT_CLOUD" ]; then
    echo "ERROR: Expected 10k Gaussian model missing:"
    echo "$POINT_CLOUD"
    exit 1
fi

echo
echo "10k Gaussian:"
ls -lh "$POINT_CLOUD"

# ============================================================
# STAGE 2 — TRAIN + TEST RGB/DEPTH RENDERING
# ============================================================

echo
echo "============================================================"
echo "STAGE 2: TRAIN + TEST RGB/DEPTH RENDERING"
echo "============================================================"

python -u render.py \
    -s "$SCENE" \
    -m "$MODEL" \
    -r 1 \
    --eval \
    --n_views 21 \
    --iteration 10000 \
    --render_depth

# ------------------------------------------------------------
# Verify renders
# ------------------------------------------------------------

TRAIN_COUNT=$(
    find "$MODEL/train" \
      -type f \
      -path "*/renders/*" \
      -name "*.png" \
      ! -name "*_depth.png" \
      2>/dev/null | wc -l
)

TEST_COUNT=$(
    find "$MODEL/test" \
      -type f \
      -path "*/renders/*" \
      -name "*.png" \
      ! -name "*_depth.png" \
      2>/dev/null | wc -l
)

echo
echo "============================================================"
echo "RENDER CHECK"
echo "============================================================"
echo "Train renders: $TRAIN_COUNT"
echo "Test renders : $TEST_COUNT"

echo
echo "Test files:"
find "$MODEL/test" \
    -type f \
    -path "*/renders/*" \
    -name "*.png" \
    -print 2>/dev/null | sort

if [ "$TRAIN_COUNT" -ne 21 ]; then
    echo "ERROR: Expected 21 train renders."
    exit 1
fi

if [ "$TEST_COUNT" -ne 3 ]; then
    echo "ERROR: Expected 3 test renders."
    exit 1
fi

echo
echo "============================================================"
echo "SPARSESURF SMALLROOM 21-VIEW STABLE POST-7K V1 10K SUCCESS"
echo "============================================================"
echo "Gaussian: $POINT_CLOUD"
echo "Train renders: 21"
echo "Test renders: 3"
echo "============================================================"
