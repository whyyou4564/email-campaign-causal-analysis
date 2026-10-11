"""仅从已有 ZIP 导入公开原始 CSV；不会解压其他项目源码。"""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import LOCAL_SHA256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/data/raw/hillstrom.csv")]
        if len(candidates) != 1:
            raise ValueError("必须有且仅有一份 data/raw/hillstrom.csv。")
        raw = archive.read(candidates[0])
    if hashlib.sha256(raw).hexdigest() != LOCAL_SHA256:
        raise ValueError("原始 CSV 校验值与已核对版本不一致。")
    destination = ROOT / "data/raw/hillstrom.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    print("Imported raw data only:", destination)

if __name__ == "__main__":
    main()
