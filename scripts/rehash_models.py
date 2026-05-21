"""
scripts/rehash_models.py

기존 모델 파일(.pkl)에 SHA-256 해시 사이드카(.sha256)를 소급 생성한다.
TASK-12 보안 강화 이전에 저장된 모델을 integrity 검증 체계에 편입시킨다.

실행:
    python scripts/rehash_models.py
    python scripts/rehash_models.py --models-dir path/to/models
"""

import argparse
import sys
from pathlib import Path

# src/ 를 경로에 추가
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from prediction._integrity import save_hash


def rehash_all(models_dir: Path) -> None:
    pkl_files = sorted(models_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {models_dir}")
        return

    for pkl in pkl_files:
        sidecar = Path(str(pkl) + ".sha256")
        if sidecar.exists():
            print(f"[SKIP] {pkl.name} — hash already exists")
            continue
        save_hash(pkl)
        print(f"[OK]   {pkl.name} → {sidecar.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill SHA-256 hashes for model files.")
    parser.add_argument(
        "--models-dir",
        default=str(_ROOT / "models"),
        help="Directory containing .pkl model files (default: models/)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: directory not found: {models_dir}")
        sys.exit(1)

    print(f"Rehashing models in: {models_dir}")
    rehash_all(models_dir)
    print("Done.")


if __name__ == "__main__":
    main()
