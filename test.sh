#!/bin/bash

echo "✅ Starting test..."

# Set default values if not provided. Paths are set for the target machine.
EPOCH_TIME=${epoch_time:-249}
CFG=${cfg:-"D:/Bharat/O-Net-main/configs/swin_tiny_patch4_window7_224_lite.yaml"}
DATA_DIR=${data_dir:-"D:/Bharat/O-Net-main/data/Synapse/test_vol_h5"}
LIST_DIR=${list_dir:-"D:/Bharat/O-Net-main/lists/lists_Synapse"}
IMG_SIZE=${img_size:-224}
BATCH_SIZE=${batch_size:-24}
OUT_DIR=${out_dir:-"D:/Bharat/O-Net-main/model_out"}

# Define the full path to the python executable, using forward slashes for bash compatibility.
PYTHON_PATH="C:/Users/HP/anaconda3/envs/onet/python.exe"

# Run the test using the full path
echo "Running python script..."
"$PYTHON_PATH" test_o_net.py \
  --dataset Synapse \
  --cfg "$CFG" \
  --is_savenii \
  --volume_path "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --list_dir "$LIST_DIR" \
  --epochs_numbers $EPOCH_TIME \
  --img_size $IMG_SIZE \
  --batch_size $BATCH_SIZE

echo "✅ Test finished."