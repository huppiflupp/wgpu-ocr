#!/bin/sh
# Fetch the ocrs ONNX models (CC-BY-SA-4.0, by Robert Knight) into ./models and check them.
# The same files and SHA-256 sums as PdfCraft's ATTRIBUTION.toml (storytold/pdfcraft#4969).
set -eu
cd "$(dirname "$0")/.."
mkdir -p models
fetch() {
  [ -f "models/$1" ] || curl -fL -o "models/$1" "https://ocrs-models.s3-accelerate.amazonaws.com/$1"
  echo "$2  models/$1" | sha256sum -c -
}
fetch text-detection.onnx a917b23dbd9524b465df7e922641b3ff2981623df4ded5a0234004ef2fee7cfe
fetch text-recognition.onnx 86c145c2edb96c8caed5b1ebb8f44d706408922211451c309c625157dd6061c5
