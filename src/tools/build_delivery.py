"""生成源码、原始数据、真实结果的交付 ZIP；排除虚拟环境、Git 和缓存。"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRECTORIES = {".venv", ".git", "__pycache__", ".pytest_cache", "work", ".cache", ".ipynb_checkpoints", "backups"}
ZIP_PATH = ROOT.parent / "email-campaign-causal-analysis-complete.zip"

def included(path):
    relative = path.relative_to(ROOT)
    return not any(part in EXCLUDED_DIRECTORIES for part in relative.parts)

def main():
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and included(path) and path.name != "delivery-manifest.json"
    )
    manifest = {
        "project": "Email Campaign A/B Testing and Uplift Modeling",
        "intended_install_directory": "E:/项目2（AB）",
        "packaging_note": "Local delivery includes original CSV and generated outputs; .gitignore excludes raw data from publication.",
        "excluded": sorted(EXCLUDED_DIRECTORIES),
        "files": [
            {"path": path.relative_to(ROOT).as_posix(),
             "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in files
        ],
    }
    manifest_path = ROOT / "reports/delivery-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(manifest_path)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        for entry in manifest["files"]:
            blob = archive.read(entry["path"])
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                raise ValueError("Manifest checksum mismatch: " + entry["path"])
    print(json.dumps({"archive": str(ZIP_PATH), "files": len(files), "bytes": ZIP_PATH.stat().st_size}, ensure_ascii=False))

if __name__ == "__main__":
    main()
