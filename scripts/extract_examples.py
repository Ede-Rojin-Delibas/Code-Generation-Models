#Preparing prompt/completion pairs from raw dataset
import json
from pathlib import Path

"""
extract_examples.py

Bu script:
- data/raw/ altındaki Python dosyalarını okur
- Her dosyayı bir instruction-style prompt + completion örneğine dönüştürür
- Sonucu JSONL formatında data/processed/ klasörüne kaydeder
"""

# -----------------------------
# CONFIG
# -----------------------------
RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/train.jsonl")

PROMPT_TEMPLATE = """### Task:
Write a Python function that matches the following code.

### Answer:
"""

VALID_EXTENSIONS = {".py"}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def read_code_file(file_path: Path) -> str:
    """
    Bir kod dosyasını güvenli şekilde okur.
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def build_example(code: str,filename: str) -> dict:
    """
    Ham kodu prompt-completion formatına dönüştürür.
    """
    return {
        "prompt": f"""### Task:
Write a Python function similar to `{filename}`.

### Answer:
""",
        "completion": code.strip()
    }


# -----------------------------
# MAIN LOGIC
# -----------------------------
def main():
    examples = []

    print(f"Scanning raw data directory: {RAW_DATA_DIR}")

    for file_path in RAW_DATA_DIR.rglob("*"):
        if file_path.suffix not in VALID_EXTENSIONS:
            continue

        code = read_code_file(file_path)

        if not code or len(code.strip()) < 20:
            continue  # çok kısa veya boş dosyaları atla

        example = build_example(code,file_path.name)
        examples.append(example)

    print(f"Total examples extracted: {len(examples)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
