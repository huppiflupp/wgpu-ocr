"""OCR engines on the same pages: time per page (after a warm-up) and error rates against
the known text. Usage: bench.py ENGINE OUTDIR PAGE.png...

ENGINE: tesseract | paddleocr | paddleocr-gpu | rapidocr-cpu | rapidocr-gpu | score
score: OUTDIR holds <engine>_<k>.txt files; prints CER/WER per engine against TRUTH
(default data/truth_A.txt). PaddleOCR on the CPU runs without MKL-DNN, which fails in
Paddle 3.3 (so its CPU time is not representative); paddleocr-gpu needs paddlepaddle-gpu.
"""

import os
import subprocess
import sys
import time


def run_tesseract(pages):
    texts, times = [], []
    subprocess.run(["tesseract", pages[0], "-", "-l", "eng"], capture_output=True, check=True)
    for p in pages:
        t = time.perf_counter()
        r = subprocess.run(["tesseract", p, "-", "-l", "eng"], capture_output=True, check=True, text=True)
        times.append(time.perf_counter() - t)
        texts.append(r.stdout)
    return texts, times


def run_paddleocr(pages, device="cpu"):
    from paddleocr import PaddleOCR

    extra = {"enable_mkldnn": False} if device == "cpu" else {}
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
        **extra,
    )
    ocr.predict(pages[0])
    texts, times = [], []
    for p in pages:
        t = time.perf_counter()
        res = ocr.predict(p)
        times.append(time.perf_counter() - t)
        texts.append("\n".join(r for page in res for r in page["rec_texts"]))
    return texts, times


def run_rapidocr(pages, gpu):
    from rapidocr import RapidOCR

    params = {}
    if gpu:
        params.update({"EngineConfig.onnxruntime.use_cuda": True})
    if gpu:
        import onnxruntime
        onnxruntime.preload_dlls()
    engine = RapidOCR(params=params)
    engine(pages[0])
    texts, times = [], []
    for p in pages:
        t = time.perf_counter()
        res = engine(p)
        times.append(time.perf_counter() - t)
        texts.append("\n".join(res.txts or []))
    return texts, times


def score(outdir):
    from rapidfuzz.distance import Levenshtein

    here = os.path.dirname(os.path.abspath(__file__))
    truth = " ".join(open(os.environ.get("TRUTH", os.path.join(here, "..", "data", "truth_A.txt"))).read().split())
    engines = sorted({f.rsplit("_", 1)[0] for f in os.listdir(outdir) if f.endswith(".txt")})
    print(f"{'engine':<16} {'CER %':>7} {'WER %':>7} {'words':>6}")
    for e in engines:
        files = sorted((f for f in os.listdir(outdir) if f.startswith(e + "_") and f.endswith(".txt")), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
        text = " ".join(" ".join(open(os.path.join(outdir, f)).read().split()) for f in files)
        cer = Levenshtein.distance(text, truth) / len(truth)
        wer = Levenshtein.distance(text.split(), truth.split()) / len(truth.split())
        print(f"{e:<16} {100 * cer:7.2f} {100 * wer:7.2f} {len(text.split()):6}")


def main():
    engine, outdir, pages = sys.argv[1], sys.argv[2], sys.argv[3:]
    if engine == "score":
        return score(outdir)
    runners = {
        "tesseract": run_tesseract,
        "paddleocr": run_paddleocr,
        "paddleocr-gpu": lambda p: run_paddleocr(p, "gpu:0"),
        "rapidocr-cpu": lambda p: run_rapidocr(p, False),
        "rapidocr-gpu": lambda p: run_rapidocr(p, True),
    }
    texts, times = runners[engine](pages)
    os.makedirs(outdir, exist_ok=True)
    for k, t in enumerate(texts):
        with open(os.path.join(outdir, f"{engine}_{k}.txt"), "w") as f:
            f.write(t)
    print(f"{engine}: {1000 * sum(times) / len(times):.0f} ms a page over {len(times)} pages ({', '.join(f'{1000 * t:.0f}' for t in times)})")


if __name__ == "__main__":
    main()
