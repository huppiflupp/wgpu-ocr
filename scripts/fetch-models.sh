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

# PP-OCRv6 small recognition and detection (PaddlePaddle, Apache-2.0) and the dictionary, for
# REC=ppocr, DET=ppocr and
# ppocr-bench: the same files as storytold/pdfcraft#1012.
mkdir -p models/ppocr
fetch_url() {
  [ -f "models/ppocr/$1" ] || curl -fL -o "models/ppocr/$1" "$2"
  echo "$3  models/ppocr/$1" | sha256sum -c -
}
fetch_url PP-OCRv6_small_rec.onnx https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.onnx 5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634
fetch_url PP-OCRv6_small_det.onnx https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/28fe5895c24fd108c19eb3e8479f4ab385fbfc62/inference.onnx d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e
fetch_url ppocrv6_dict.txt https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/ppocr/utils/dict/ppocrv6_dict.txt b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d
