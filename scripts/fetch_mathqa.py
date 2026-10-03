"""Download the fixed upstream MathQA archive and check its SHA256."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
EXPECTED = "7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a"


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    destination = ROOT / "optional/extensions_v2/assets/MathQA.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != EXPECTED:
            raise ValueError("Existing MathQA archive has a different checksum.")
        print("MathQA archive already matches the recorded source.")
        return
    temporary = destination.with_suffix(".zip.download")
    request = urllib.request.Request(URL, headers={"User-Agent": "math-reasoning-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(chunk)
    if file_hash(temporary) != EXPECTED:
        raise ValueError("Downloaded archive differs from the recorded SHA256; check the upstream source.")
    temporary.replace(destination)
    print("MathQA source verified:", EXPECTED)


if __name__ == "__main__":
    main()
