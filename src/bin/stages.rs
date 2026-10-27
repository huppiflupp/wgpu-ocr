//! Whole-page OCR with ocrs on the CPU (today's path) against the same pages with both models on
//! the GPU, weights read from the ONNX files in Rust. ocrs finds the lines and prepares the line
//! images in both; the GPU path batches them as ocrs does (width rounded up to 50, at most 20
//! lines a batch). Reports, per stage, the time on each path, and whether the text matches.

use wgpu_ocr::{gpu, onnx};

use std::collections::BTreeMap;
use std::time::Instant;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::FloatOperators;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

const ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const BLACK: f32 = -0.5;

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// Greedy CTC over `steps` time steps of one line in [t][ntot][classes]. As ocrs does, a
/// character that starts in the padding (step x 4 at or past the line's own width) is dropped.
fn decode(out: &[f32], line: usize, ntot: usize, steps: usize, classes: usize, width: usize) -> String {
    let alphabet: Vec<char> = ALPHABET.chars().collect();
    let (mut s, mut prev) = (String::new(), 0);
    for t in 0..steps {
        let row = &out[(t * ntot + line) * classes..(t * ntot + line + 1) * classes];
        let best = (0..classes).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap_or(0);
        if best != 0 && best != prev && t * 4 < width {
            s.push(alphabet[best - 1]);
        }
        prev = best;
    }
    s
}

fn words(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Default)]
struct Times {
    prep: f64,
    detect_words: f64,
    detect_pixels: f64,
    find_lines: f64,
    recognize: f64,
    // GPU path
    resize_in: f64,
    det_gpu: f64,
    det_cpu_model: f64,
    resize_out: f64,
    line_prep: f64,
    rec_gpu: f64,
    decode: f64,
}

fn main() -> Result<(), String> {
    let pages: Vec<String> = std::env::args().skip(1).collect();
    let models = std::env::var("MODELS").unwrap_or_else(|_| "models".into());
    let lines_per_wg: usize = std::env::var("LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let (det_path, rec_path) = (format!("{models}/text-detection.onnx"), format!("{models}/text-recognition.onnx"));

    let start = Instant::now();
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(rten::Model::load_file(&det_path).map_err(|e| e.to_string())?),
        recognition_model: Some(rten::Model::load_file(&rec_path).map_err(|e| e.to_string())?),
        ..Default::default()
    })
    .map_err(|e| e.to_string())?;
    let det_cpu = rten::Model::load_file(&det_path).map_err(|e| e.to_string())?;
    println!("ocrs (CPU) loaded in {:.0} ms", ms(start));

    let start = Instant::now();
    let gpu = gpu::Gpu::new(lines_per_wg)?;
    let (det_onnx, rec_onnx) = (onnx::Onnx::load(&det_path)?, onnx::Onnx::load(&rec_path)?);
    let detector = gpu::Detector::new(&gpu, &det_onnx)?;
    let recognizer = gpu::Recognizer::new(&gpu, &rec_onnx, lines_per_wg)?;
    println!("GPU ({}) loaded from ONNX in {:.0} ms", gpu.adapter, ms(start));

    let threshold = engine.detection_threshold();
    let page_batch: usize = std::env::var("PAGEBATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    #[allow(clippy::type_complexity)]
    let mut pending: Vec<(usize, Vec<(Vec<f32>, usize, usize)>, Vec<usize>, Vec<usize>, Vec<String>, usize)> = Vec::new();
    let mut total = Times::default();
    let (mut same, mut lines_total, mut det_diff, mut det_agree, mut det_px, mut det_text) = (0, 0, 0f32, 0usize, 0usize, 0usize);
    // The first page once more up front, untimed: shader and allocator warm-up on both paths.
    let warm = std::env::var_os("NOWARM").is_none();
    let runs: Vec<(bool, &String)> = warm.then(|| (true, &pages[0])).into_iter().chain(pages.iter().map(|p| (false, p))).collect();
    for (ri, (warmup, page)) in runs.into_iter().enumerate() {
        let pi = if warm { ri.saturating_sub(1) } else { ri };
        let img = image::open(page).map_err(|e| e.to_string())?.into_rgb8();
        let (w, h) = img.dimensions();
        let mut t = Times::default();

        // --- ocrs on the CPU, stage by stage.
        let s = Instant::now();
        let input = engine.prepare_input(ImageSource::from_bytes(img.as_raw(), (w, h)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        t.prep = ms(s);
        let s = Instant::now();
        let word_rects = engine.detect_words(&input).map_err(|e| e.to_string())?;
        t.detect_words = ms(s);
        let s = Instant::now();
        let _ = engine.detect_text_pixels(&input).map_err(|e| e.to_string())?;
        t.detect_pixels = ms(s);
        let s = Instant::now();
        let line_rects = engine.find_text_lines(&input, &word_rects);
        t.find_lines = ms(s);
        let s = Instant::now();
        let cpu_text: Vec<String> = engine
            .recognize_text(&input, &line_rects)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|l| l.map(|l| l.to_string()).unwrap_or_default())
            .collect();
        t.recognize = ms(s);

        // --- Detection on the GPU: the same input ocrs gives its model (grey, resized to the
        // model's 800 x 600), checked against rten running the model on that input.
        let s = Instant::now();
        let grey: Vec<f32> = img.pixels().map(|p| BLACK + (p[0] as f32 * 0.299 + p[1] as f32 * 0.587 + p[2] as f32 * 0.114) / 255.0).collect();
        let page_t = Tensor::from_data(&[1, 1, h as usize, w as usize], grey);
        let resized = page_t.resize_image([800, 600]).map_err(|e| e.to_string())?;
        let det_in: Vec<f32> = resized.to_vec();
        t.resize_in = ms(s);
        let s = Instant::now();
        let mask = detector.detect(&gpu, &det_in, 800, 600)?;
        t.det_gpu = ms(s);
        let s = Instant::now();
        let reference: Tensor<f32> = det_cpu.run_one(resized.view().into(), None).map_err(|e| e.to_string())?.try_into().map_err(|e| format!("{e:?}"))?;
        t.det_cpu_model = ms(s);
        if warmup {
            // Everything below runs once untimed and is not counted.
        }
        det_text += if warmup { 0 } else { mask.iter().filter(|&&v| v > threshold).count() };
        for (a, b) in mask.iter().zip(reference.iter()) {
            if warmup {
                break;
            }
            det_diff = det_diff.max((a - b).abs());
            det_agree += usize::from((*a > threshold) == (*b > threshold));
        }
        if !warmup {
            det_px += mask.len();
        }
        let s = Instant::now();
        let back = Tensor::from_data(&[1, 1, 800, 600], mask).resize_image([h as usize, w as usize]).map_err(|e| e.to_string())?;
        drop(back);
        t.resize_out = ms(s);

        // --- Recognition on the GPU: ocrs' own line images, batched as ocrs batches them.
        let s = Instant::now();
        // The line images in parallel, as ocrs' recognize_text prepares them per batch in parallel.
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = line_rects.len().div_ceil(threads).max(1);
        let prepared: Vec<Result<(usize, NdTensor<f32, 2>), String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = line_rects
                .chunks(chunk)
                .enumerate()
                .map(|(c, lines)| {
                    let (engine, input) = (&engine, &input);
                    scope.spawn(move || {
                        lines
                            .iter()
                            .enumerate()
                            .filter(|(_, l)| !l.is_empty())
                            .map(|(k, l)| engine.prepare_recognition_input(input, l).map(|img| (c * chunk + k, img)).map_err(|e| e.to_string()))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });
        let mut groups: BTreeMap<usize, Vec<(usize, NdTensor<f32, 2>)>> = BTreeMap::new();
        for r in prepared {
            let (i, img) = r?;
            let width = img.size(1).next_multiple_of(50);
            groups.entry(width).or_default().push((i, img));
        }
        let (mut batches, mut order, mut widths) = (Vec::new(), Vec::new(), Vec::new());
        for (width, members) in &groups {
            for chunk in members.chunks(20) {
                let mut data = vec![BLACK; chunk.len() * 64 * width];
                for (k, (i, img)) in chunk.iter().enumerate() {
                    let lw = img.size(1);
                    for y in 0..64 {
                        for x in 0..lw {
                            data[(k * 64 + y) * width + x] = img[[y, x]];
                        }
                    }
                    order.push(*i);
                    widths.push(lw);
                }
                batches.push((data, chunk.len(), *width));
            }
        }
        t.line_prep = ms(s);
        if warmup {
            let _ = recognizer.recognize(&gpu, &batches)?;
            continue;
        }
        for (a, b) in [
            (&mut total.prep, t.prep),
            (&mut total.detect_words, t.detect_words),
            (&mut total.detect_pixels, t.detect_pixels),
            (&mut total.find_lines, t.find_lines),
            (&mut total.recognize, t.recognize),
            (&mut total.resize_in, t.resize_in),
            (&mut total.det_gpu, t.det_gpu),
            (&mut total.det_cpu_model, t.det_cpu_model),
            (&mut total.resize_out, t.resize_out),
            (&mut total.line_prep, t.line_prep),
        ] {
            *a += b;
        }
        // The lines of PAGEBATCH pages go through one recognition: the GRU's 601 steps cost
        // about the same for one page's lines as for several pages'.
        pending.push((pi, batches, order, widths, cpu_text, line_rects.len()));
        if pending.len() < page_batch && pi + 1 < pages.len() {
            continue;
        }
        let all: Vec<(Vec<f32>, usize, usize)> = pending.iter().flat_map(|p| p.1.iter().cloned()).collect();
        let s = Instant::now();
        let (logits, _tmax, lens) = recognizer.recognize(&gpu, &all)?;
        let rec_ms = ms(s);
        total.rec_gpu += rec_ms;
        let s = Instant::now();
        let ntot: usize = pending.iter().map(|p| p.2.len()).sum();
        let mut k0 = 0;
        let mut texts = Vec::new();
        for (_, _, order, widths, _, n) in &pending {
            let mut gpu_text = vec![String::new(); *n];
            for (k, &i) in order.iter().enumerate() {
                gpu_text[i] = decode(&logits, k0 + k, ntot, lens[k0 + k], 97, widths[k]);
            }
            k0 += order.len();
            texts.push(gpu_text);
        }
        total.decode += ms(s);
        for ((pi, batches, _, _, cpu_text, n), gpu_text) in pending.drain(..).zip(texts) {
            let page_same = gpu_text.iter().zip(&cpu_text).filter(|(a, b)| words(a) == words(b)).count();
            for (i, (a, b)) in gpu_text.iter().zip(&cpu_text).enumerate().filter(|(_, (a, b))| words(a) != words(b)).take(3) {
                println!("  page {pi} line {i} differs:\n    cpu: {b}\n    gpu: {a}");
            }
            same += page_same;
            lines_total += cpu_text.len();
            println!("page {pi}: {n} lines in {} batches, same text {page_same}/{}", batches.len(), cpu_text.len());
        }
        println!("  recognition of these pages on the GPU: {rec_ms:.0} ms");
    }
    let t = &total;
    let post = t.detect_words - t.detect_pixels;
    let cpu = t.prep + t.detect_words + t.find_lines + t.recognize;
    let gpu_path = t.prep + t.resize_in + t.det_gpu + t.resize_out + post + t.find_lines + t.line_prep + t.rec_gpu + t.decode;
    println!("\n== {} pages, {lines_total} lines: same text {same}/{lines_total}", pages.len());
    println!("detection: {det_text} of {det_px} mask pixels are text (> {threshold})");
    println!("detection: GPU mask vs rten max |Δ| {det_diff:.5}, same side of threshold {threshold}: {:.4}%", 100.0 * det_agree as f64 / det_px.max(1) as f64);
    println!("{:<44} {:>10} {:>10}", "stage (all pages)", "CPU ms", "GPU ms");
    println!("{:<44} {:>10.0} {:>10.0}", "prepare input (ocrs)", t.prep, t.prep);
    println!("{:<44} {:>10.0} {:>10}", "detection: resize + model + resize back", t.detect_pixels, format!("{:.0}", t.resize_in + t.det_gpu + t.resize_out));
    println!("{:<44} {:>10} {:>10.0}", "  of which the model alone", format!("{:.0}", t.det_cpu_model), t.det_gpu);
    println!("{:<44} {:>10.0} {:>10.0}", "detection: words from the mask (ocrs)", post, post);
    println!("{:<44} {:>10.0} {:>10.0}", "find lines (ocrs)", t.find_lines, t.find_lines);
    println!("{:<44} {:>10.0} {:>10}", "recognition (ocrs: prep + model + decode)", t.recognize, format!("{:.0}", t.line_prep + t.rec_gpu + t.decode));
    println!("{:<44} {:>10} {:>10.0}", "  of which the model on the GPU", "", t.rec_gpu);
    println!("{:<44} {:>10.0} {:>10.0}   ({:.1}x)", "total", cpu, gpu_path, cpu / gpu_path);
    Ok(())
}
