import json
from pathlib import Path

"""
extract_examples.py

- data/raw/*.py        → manuel küçük dataset
- data/raw/github_code.jsonl → GitHub büyük dataset

Hepsini instruction-style prompt + completion'a çevirir
"""

# -----------------------------
# CONFIG
# -----------------------------
RAW_DATA_DIR = Path("data/raw")
GITHUB_JSONL = RAW_DATA_DIR / "github_code.jsonl"
OUTPUT_FILE = Path("data/processed/train.jsonl")

PROMPT_TEMPLATE = """### Instruction:
You are a helpful assistant that writes clean, correct, and efficient Python code.

### Task:
Generate the Python code.

### Answer:
"""

MIN_CODE_LENGTH = 50

# -----------------------------
# HELPERS
# -----------------------------
def build_example(code: str) -> dict:
    return {
        "prompt": PROMPT_TEMPLATE,
        "completion": code.strip()
    }


def process_py_files(examples: list):
    for file_path in RAW_DATA_DIR.rglob("*.py"):
        try:
            code = file_path.read_text(encoding="utf-8")
            if len(code.strip()) < MIN_CODE_LENGTH:
                continue
            examples.append(build_example(code))
        except Exception:
            continue


def process_github_jsonl(examples: list):
    if not GITHUB_JSONL.exists():
        return

    with GITHUB_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                code = item.get("content", "")
                if len(code.strip()) < MIN_CODE_LENGTH:
                    continue
                examples.append(build_example(code))
            except Exception:
                continue


# -----------------------------
# MAIN
# -----------------------------
def main():
    examples = []

    print("Processing local .py files...")
    process_py_files(examples)

    print("Processing GitHub JSONL dataset...")
    process_github_jsonl(examples)

    print(f"Total examples collected: {len(examples)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
