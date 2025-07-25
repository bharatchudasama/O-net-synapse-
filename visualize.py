import os
import numpy as np
import imageio
from tqdm import tqdm

# List of epoch folders to visualize
epoch_folders = [
    "test_log/epoch_199",
    "test_log/epoch_249",
    "test_log/epoch_299"
]

# Output root folder for PNG images
output_root = "visualization"

for epoch_folder in epoch_folders:
    epoch_name = os.path.basename(epoch_folder)
    npz_dir = os.path.join(epoch_folder)
    
    if not os.path.exists(npz_dir):
        print(f"❌ Skipping missing folder: {npz_dir}")
        continue

    print(f"🔍 Processing {epoch_name} ...")
    
    for fname in tqdm(os.listdir(npz_dir)):
        if not fname.endswith('.npz'):
            continue

        npz_path = os.path.join(npz_dir, fname)
        data = np.load(npz_path)

        case_id = fname.replace('.npz', '')
        case_out_dir = os.path.join(output_root, epoch_name, case_id)
        os.makedirs(case_out_dir, exist_ok=True)

        pred = data['pred']
        label = data['label'] if 'label' in data else None
        image = data['image'] if 'image' in data else None

        for i in range(pred.shape[0]):
            imageio.imwrite(os.path.join(case_out_dir, f'pred_slice{i:03d}.png'), (pred[i] * 255).astype(np.uint8))
            if label is not None:
                imageio.imwrite(os.path.join(case_out_dir, f'label_slice{i:03d}.png'), (label[i] * 255).astype(np.uint8))
            if image is not None:
                img_norm = ((image[i] - np.min(image[i])) / (np.ptp(image[i]) + 1e-8) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(case_out_dir, f'image_slice{i:03d}.png'), img_norm)

print("✅ All epoch predictions visualized!")
