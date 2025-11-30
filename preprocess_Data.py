# ...existing code...
import zipfile, os
from pathlib import Path

DATA_PATH = "Pre_trained_Data/Local_Data.npz"
OUT_DIR = "Pre_trained_Data/splitted_npy"
os.makedirs(OUT_DIR, exist_ok=True)

def stream_extract_npz(npz_path, out_dir):
    with zipfile.ZipFile(npz_path, 'r') as zf:
        members = [m for m in zf.namelist() if m.endswith('.npy')]
        print("Members to extract:", members)
        for member in members:
            name = Path(member).name
            if name not in ("observation.npy", "action_mask.npy", "action.npy"):
                continue
            out_path = Path(out_dir) / name
            print(f"Extracting {member} -> {out_path}")
            with zf.open(member, 'r') as src, open(out_path, 'wb') as dst:
                while True:
                    chunk = src.read(1 << 20)  # 1MB
                    if not chunk:
                        break
                    dst.write(chunk)
    print("Extraction done. Files in:", out_dir)

if __name__ == "__main__":
    stream_extract_npz(DATA_PATH, OUT_DIR)
# ...existing code...