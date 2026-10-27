# wgpu-ocr

The [ocrs](https://github.com/robertknight/ocrs) OCR models — text detection and text recognition — running on **wgpu compute shaders**: Vulkan, Metal, DX12, and later WebGPU. It reads the model graph and weights straight from the `.onnx` files in Rust, with no ML framework and no vendor runtime (CUDA, ROCm). It also includes a whole-page pipeline that overlaps GPU and CPU across pages.

**Status: prototype.** It was written to answer [storytold/pdfcraft#2285](https://github.com/storytold/pdfcraft/issues/2285) ("OCR should be GPU accelerated") with measurements, and it is not a finished library: errors still panic, the tuning is per device, and there is no API stability. It is published so others can reuse the kernels, the executor and the numbers.

## Results

12 pages (the 4 pages of `data/pages/A_300` three times; US Letter at 300 dpi; 432 lines). The time is per page after a warm-up. Errors are measured against the known source text (`data/truth_A.txt`).

### Speed: ocrs on the CPU against this pipeline

| Machine | ocrs on the CPU, page by page | wgpu-ocr (GPU + CPU overlapped) | Speed-up |
|---|---|---|---|
| Core Ultra 5 245K (14 cores) + RTX 5080, Vulkan | 581 ms | **85 ms** | **6.9×** |
| Ryzen AI Max+ 395 (16 cores), Radeon 8060S iGPU, RADV | 545 ms | **154 ms** | **3.5×** |

### Against other engines (Core Ultra 5 245K + RTX 5080, the 4 pages of `A_300`)

| Engine | ms per page | CER | WER |
|---|---|---|---|
| **wgpu-ocr** (ocrs models, line padding, GPU) | **85** | **0.01 %** | **0.05 %** |
| ocrs 0.13.1 on the CPU (as shipped) | 581 | 0.65 % | 4.32 % |
| Tesseract 5.5.3, one process | 1048 | 0.00 % | 0.00 % |
| Tesseract 5.5.3, 14 processes in parallel (throughput) | 128 | 0.00 % | 0.00 % |
| PaddleOCR 3.7 on the GPU (Paddle 3.4, CUDA 13) | 524 | 0.00 % | 0.00 % |
| PaddleOCR 3.7 on the CPU (MKL-DNN off, see below) | 32 800 | 0.00 % | 0.00 % |
| RapidOCR 3.10 (PP-OCRv6 small, onnxruntime-gpu), defaults | 349 | 6.95 % | 8.28 % |
| RapidOCR 3.10, onnxruntime CPU, defaults | 1836 | 7.48 % | 8.91 % |

On the Ryzen AI Max+ 395, Tesseract with 16 to 32 parallel processes reaches **110 ms per page**. That is faster than this pipeline on that iGPU (154 ms).

How to read this table:
- **The test pages are clean synthetic renders.** Real scans (noise, skew, tables) may rank the engines differently.
- **ocrs' errors come from its pipeline, not its model.** It clips the first and last letters of lines ("The" → "he", "and" → "an", full stops lost). ocrs grows word boxes by a fixed 3 px, which at 300 dpi is under one pixel of the 800 × 600 detection input. This pipeline widens each found line by 0.15 of its median word height above and below, and by 0.1 before the first and after the last word. The same fix went to PdfCraft as storytold/pdfcraft#5058. Over all validation pages (two texts at 150, 200, 300 and 400 dpi) the word error rate falls from 3.78 % to 0.30 % (`bench/sweep.sh`).
- **PaddleOCR on the CPU is not representative.** Paddle 3.3's oneDNN path fails ("ConvertPirAttribute2RuntimeAttribute not support"), so it ran with `enable_mkldnn=False`.
- **RapidOCR ran with its defaults and was not tuned.** Its errors are detection boxes between text lines.

## What is in it

- **`src/onnx.rs`.** The model graph and weights, read with `rten-onnx` (rten's ONNX parser).
- **`src/gpu.rs`.** Everything on the GPU:
  - **Graph executor** for convolutional ONNX graphs: Conv (dense as an implicit GEMM with 64- or 128-pixel × 64-channel tiles; depthwise; one input channel direct, with ReLU and MaxPool fused), ConvTranspose, MaxPool, Slice, Concat, zero Pad, and Relu/Sigmoid fused into the preceding conv. It runs the whole detection U-Net and the recognition trunk.
  - **Detector.** Takes RGB bytes in and returns a thresholded bit mask at page size. Grey conversion, padding and resize happen on the GPU, matching rten's `resize_image` (half-pixel, clamped).
  - **Recogniser.** The input projection of each GRU layer is a single GEMM over all time steps. The recurrence is **one dispatch per layer for all lines**: each workgroup walks all time steps with the hidden state in workgroup memory, and every line has its own length. The recurrent weights are in f16 when `SHADER_F16` is available, and the loads are pipelined. Then come the output layer and greedy CTC on the CPU.
  - **Buffers.** Scratch buffers are pooled, because wgpu zero-fills every new buffer.
- **`src/bin/ocr-bench.rs`.** The whole-page pipeline against ocrs:
  - The GPU runs on its own thread with a job queue. While it reads page *k*, the CPU finds the words and lines of page *k+1*.
  - Recognitions that wait are merged (`MERGE`, 2 pages is best).
  - Words come from the mask with ocrs' algorithm (contours, simplify, minimum-area rectangle; see NOTICE).
  - Lines are found by ocrs, and the line images are cut by ocrs and batched as ocrs batches them.
- **`src/bin/stages.rs`.** The same, stage by stage, with times per stage and a check of the detection mask against rten.

### Correctness

- **Detection.** On the same input, the mask matches rten to < 1e-5, and every pixel lands on the same side of the 0.2 threshold.
- **Recognition.** With the same line images as ocrs, the text matches ocrs on every line, including ocrs' rule that drops characters decoded in the right-hand padding.
- **Word count.** The pipeline finds the same number of words as ocrs on every page.

### Where the time goes, and what did not help

- **The GPU is the limit on both machines.** The CPU is busy about 45 ms per page.
- **Convolutions** reach about 8 TFLOPS on the RTX 5080, a fraction of peak. Larger register tiles and vec4 shared-memory loads did not help. On RDNA 3.5 they are slower.
- **The GRU recurrence** (601 sequential steps) is latency-bound on streaming its weights through one compute unit per step. 768 threads per workgroup and more lines per workgroup did not help, while f16 weights and pipelined loads did.
- **Giving part of the recognition to the CPU** (`CPUSHARE`) once helped on the iGPU (158 → 151 ms) but was not reproducible, so it is off by default.
- **Deliberately not used:** a "persistent" kernel with spin barriers across workgroups. WebGPU guarantees neither visibility nor forward progress for that, so it could hang the GPU.
- **Next steps would be:** matrix units (tensor cores / WMMA, experimental in wgpu) and subgroup reductions.

## Reproduce

```sh
scripts/fetch-models.sh          # ocrs ONNX models (CC-BY-SA-4.0), SHA-256 checked
cargo build --release
target/release/ocr-bench $(ls data/pages/A_300/*.png | sort -V)
REPEAT=3 MERGE=2 target/release/ocr-bench $(ls data/pages/A_300/*.png | sort -V)
```

Settings, all optional environment variables:

| Variable | Meaning | RTX 5080 | Radeon 8060S |
|---|---|---|---|
| `LINES` | lines per GRU workgroup | 4 | 8 |
| `RNN` | 768 (one thread per weight row) or 256 | 768 | 768 |
| `NO128` | use the 64-pixel conv tile instead of 128 | – | 1 |
| `F16` | 0 keeps the GRU weights in f32 | 1 | 1 |
| `UNROLL` | GRU weight loads per block | 4 | 4 |
| `MERGE` | pages per recognition pass | 2 | 2 |
| `CPUSHARE` | share of recognition batches given to rten on the CPU | 0 | 0 |
| `RELV`, `RELEND` | line padding as fractions of word height | 0.15, 0.1 | 0.15, 0.1 |
| `REPEAT` | run the pages this many times | | |
| `DUMP` | directory to write the text of each page to | | |

Comparing with other engines (Python 3.12, `pip install -r bench/requirements.txt`):

```sh
python bench/bench.py tesseract results/run data/pages/A_300/*.png
python bench/bench.py paddleocr-gpu results/run data/pages/A_300/*.png   # needs paddlepaddle-gpu
python bench/bench.py score results/run                                  # CER/WER against data/truth_A.txt
PY=python bench/sweep.sh "RELV=0 RELEND=0" "RELV=0.15 RELEND=0.1"         # line padding over all pages
```

The test pages were rendered from `data/truth_A.txt` and `data/truth_B.txt` with PdfCraft (`data/make-pages.json` is the `pdfcraft-cli run` script).

## License

MIT OR Apache-2.0, at your option. See NOTICE for the ocrs-derived function and for the models, which are not included.
