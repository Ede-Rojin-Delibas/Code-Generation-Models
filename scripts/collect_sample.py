import json
import random
from pathlib import Path

"""
collect_sample.py

Bu script, büyük bir veri kaynağından (JSONL, API veya dosya) küçük bir örnek dataset toplamak için tasarlanmıştır.
Senin fine-tuning sürecinde, ham veri çok büyükse ve hızlıca bir preview seti istiyorsan bu script işini görür.
"""

# -----------------------------
# CONFIG
# -----------------------------
DATA_SOURCE = "data/raw/raw_dataset.jsonl"   # Ham veri dosyası
OUTPUT_PATH = "data/processed/sample_dataset.jsonl"  # Örnek veri dosyası
SAMPLE_SIZE = 200  # Kaç örnek alınacak
SEED = 42

random.seed(SEED)

# -----------------------------
# FUNCTIONS
# -----------------------------
def load_jsonl(path):
    """JSONL dosyasını satır satır okuyup liste döner."""
    dataset = []
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data source not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    return dataset


def save_jsonl(data, path):
    """Listeyi JSONL olarak kaydeder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved sample dataset → {path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading dataset...")
    dataset = load_jsonl(DATA_SOURCE)

    print(f"Total samples in raw data: {len(dataset)}")

    if SAMPLE_SIZE > len(dataset):
        raise ValueError("Sample size is larger than dataset size.")

    print(f"Sampling {SAMPLE_SIZE} items...")
    sample = random.sample(dataset, SAMPLE_SIZE)

    print("Saving sample dataset...")
    save_jsonl(sample, OUTPUT_PATH)

    print("Done!")


if __name__ == "__main__":
    main()
