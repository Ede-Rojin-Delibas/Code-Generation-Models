import json
import time
import requests
from pathlib import Path

"""
fetch_github_content.py

Bu script:
- BigQuery'den gelen repo_name + path içeren JSON'u okur
- GitHub RAW URL üzerinden dosya içeriğini çeker
- Başarılı olanları yeni bir JSONL dosyasına yazar

Rate-limit safe, retry'lı ve log'ludur.
"""

# -----------------------------
# CONFIG
# -----------------------------
INPUT_JSON = Path("data/raw/phase2dataset_query.json")
OUTPUT_JSONL = Path("data/raw/github_code.jsonl")

BASE_RAW_URL = "https://raw.githubusercontent.com"

REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = 1.0   # saniye
MAX_RETRIES = 3

MIN_CODE_LENGTH = 50
MAX_CODE_LENGTH = 8000

# -----------------------------
# HELPERS
# -----------------------------
def build_raw_url(repo_name: str, path: str) -> str:
    """
    GitHub raw URL oluşturur.
    Varsayılan branch: master → main fallback
    """
    return f"{BASE_RAW_URL}/{repo_name}/master/{path}"


def fetch_code(url: str) -> str | None:
    """
    Raw GitHub URL'den dosya içeriğini çeker (retry'lı).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                return response.text

            elif response.status_code == 404:
                return None  # dosya yok

        except requests.RequestException:
            pass

        time.sleep(1)

    return None


def is_valid_code(code: str) -> bool:
    """
    Basit kalite filtresi
    """
    if not code:
        return False

    if len(code) < MIN_CODE_LENGTH or len(code) > MAX_CODE_LENGTH:
        return False

    keywords = ["def ", "class ", "import "]
    return any(k in code for k in keywords)

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    success = 0
    skipped = 0

    print("Starting GitHub raw content fetch...")

    with INPUT_JSON.open("r", encoding="utf-8") as fin:
        content = fin.read().strip()

        # JSON array mi?
        if content.startswith("["):
            items = json.loads(content)

        # JSONL mi?
        else:
            items = [json.loads(line) for line in content.splitlines() if line.strip()]

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        for item in items:

            total += 1

            repo = item.get("repo_name")
            path = item.get("path")

            if not repo or not path:
                skipped += 1
                continue

            raw_url = build_raw_url(repo, path)
            code = fetch_code(raw_url)

            if not is_valid_code(code):
                skipped += 1
                continue

            output_item = {
                "repo_name": repo,
                "path": path,
                "content": code
            }

            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            success += 1

            if success % 50 == 0:
                print(f"Fetched {success} files...")

            time.sleep(SLEEP_BETWEEN_REQUESTS)


    print("Fetch finished.")
    print(f"Total entries read: {total}")
    print(f"Successfully fetched: {success}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
