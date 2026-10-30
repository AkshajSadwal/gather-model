#!/usr/bin/env bash
set -euo pipefail

# Download the two KuaiRand variants used by this project (Pure and 1K).
# Official Zenodo MD5 checksums are verified before extraction.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$PROJECT_DIR/data/raw"
BASE_URL="https://zenodo.org/records/10439422/files"
mkdir -p "$RAW_DIR"

usage() {
  echo "Usage: $0 [all|pure|1k]"
  echo "  all  Download and extract Pure + 1K (default)"
  echo "  pure Download and extract KuaiRand-Pure only"
  echo "  1k   Download and extract KuaiRand-1K only"
}

verify_file() {
  local archive="$1"
  local expected="$2"
  local actual
  actual="$(md5 -q "$archive")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: MD5 mismatch for $archive" >&2
    echo "Expected: $expected" >&2
    echo "Actual:   $actual" >&2
    return 1
  fi
}

download_and_extract() {
  local name="$1"
  local expected_md5="$2"
  local required_file="$3"
  local archive="$RAW_DIR/$name.tar.gz"
  local partial="$archive.part"
  local url="$BASE_URL/$name.tar.gz"

  if [[ -f "$RAW_DIR/$name/data/$required_file" ]]; then
    echo "Already extracted: $RAW_DIR/$name/data"
  else
    if [[ -f "$archive" ]]; then
      if verify_file "$archive" "$expected_md5"; then
        echo "Verified existing archive: $archive"
      else
        rm -f "$archive"
      fi
    fi

    if [[ ! -f "$archive" ]]; then
      echo "Downloading $name (this may take a while)..."
      if [[ -f "$partial" ]]; then
        curl -fL --retry 5 --retry-delay 3 -C - -o "$partial" "$url"
      else
        curl -fL --retry 5 --retry-delay 3 -o "$partial" "$url"
      fi
      verify_file "$partial" "$expected_md5"
      mv "$partial" "$archive"
    fi

    echo "Extracting $name..."
    tar -xzf "$archive" -C "$RAW_DIR"
  fi

  if [[ ! -f "$RAW_DIR/$name/data/$required_file" ]]; then
    echo "ERROR: Expected file not found after extraction:" >&2
    echo "  $RAW_DIR/$name/data/$required_file" >&2
    echo "Inspect the extracted folder and share its listing." >&2
    return 1
  fi
  echo "Ready: $RAW_DIR/$name/data"
}

choice="${1:-all}"
case "$choice" in
  all)
    download_and_extract "KuaiRand-Pure" "0820331067a3784d9691136f772b35a7" "log_standard_4_08_to_4_21_pure.csv"
    download_and_extract "KuaiRand-1K" "6b0b9c8222d67fcd4c676218edca3f1f" "log_standard_4_08_to_4_21_1k.csv"
    ;;
  pure)
    download_and_extract "KuaiRand-Pure" "0820331067a3784d9691136f772b35a7" "log_standard_4_08_to_4_21_pure.csv"
    ;;
  1k)
    download_and_extract "KuaiRand-1K" "6b0b9c8222d67fcd4c676218edca3f1f" "log_standard_4_08_to_4_21_1k.csv"
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "Dataset download/extraction complete."
