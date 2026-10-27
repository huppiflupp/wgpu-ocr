//! Whole-document OCR with the GPU doing everything it can, overlapped with the CPU across
//! pages, against ocrs on the CPU page by page (as PdfCraft runs it today).
//!
//! GPU thread (a job queue): detection from the page's RGB bytes to the thresholded full-size
//! mask (grey, pad, resize, model, cut, resize back, threshold), and recognition; recognition
//! jobs waiting together are merged into one pass. CPU: words from the mask (contours), lines,
//! line images, while the GPU works on the previous page.

use wgpu_ocr::{gpu, onnx};

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Instant;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten_imageproc::{RetrievalMode, RotatedRect, find_contours, min_area_rect, simplify_polygon};
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

const ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const BLACK: f32 = -0.5;

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// The words of a binary mask: ocrs' `find_connected_component_rects` (ocrs 0.13.1,
/// src/detection.rs, MIT OR Apache-2.0), with ocrs' parameters (expand by 3, min area 100).
fn word_rects(mask: &[u32], w: usize, h: usize) -> Vec<RotatedRect> {
    let row_words = w.div_ceil(32);
    let bits: Vec<bool> = (0..h * w).map(|i| (mask[(i / w) * row_words + (i % w) / 32] >> ((i % w) % 32)) & 1 == 1).collect();
    let mask = NdTensor::from_data([h, w], bits);
    // ocrs expands by 3 px; EXPAND / EXPANDX override it (diagnosis of cut-off line edges).
    let expand_dist: f32 = std::env::var("EXPAND").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0);
    let expand_x: f32 = std::env::var("EXPANDX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let min_area = 100.0f32;
    find_contours(mask.view(), RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
            let simplified = simplify_polygon(&float_points, 2.);
            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(rect.width() + 2. * (expand_dist + expand_x), rect.height() + 2. * expand_dist);
                rect
            })
        })
        .filter(|r| r.area() >= min_area)
        .collect()
}

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

struct RecPage {
    page: usize,
    batches: Vec<(Vec<f32>, usize, usize)>,
    order: Vec<usize>,
    widths: Vec<usize>,
    lines: usize,
    /// No page follows: recognize what is waiting without waiting for more.
    last: bool,
}

enum Job {
    Detect { page: usize, rgb: std::sync::Arc<Vec<u8>>, w: usize, h: usize },
    Recognize(RecPage),
}

enum Done {
    Mask(usize, Result<Vec<u32>, String>),
    /// Recognized lines of a page, by line index: one part from the GPU, one from the CPU.
    Part(usize, Result<Vec<(usize, String)>, String>),
}

/// The CPU's share of the recognition: rten on the batches the GPU doesn't get.
fn cpu_thread(jobs: mpsc::Receiver<RecPage>, done: mpsc::Sender<Done>, models: String) {
    let model = match rten::Model::load_file(format!("{models}/text-recognition.onnx")) {
        Ok(m) => m,
        Err(e) => {
            for j in jobs {
                let _ = done.send(Done::Part(j.page, Err(e.to_string())));
            }
            return;
        }
    };
    for job in jobs {
        let result = (|| -> Result<Vec<(usize, String)>, String> {
            let mut out = Vec::new();
            let mut k = 0;
            for (data, n, w) in &job.batches {
                let input = NdTensor::from_data([*n, 1, 64, *w], data.clone());
                let logits: NdTensor<f32, 3> = model.run_one(input.view().into(), None).map_err(|e| e.to_string())?.try_into().map_err(|e| format!("{e:?}"))?;
                let t = logits.size(0);
                let flat = logits.to_vec();
                for line in 0..*n {
                    out.push((job.order[k], decode(&flat, line, *n, t, 97, job.widths[k])));
                    k += 1;
                }
            }
            Ok(out)
        })();
        let _ = done.send(Done::Part(job.page, result));
    }
}

/// The GPU thread: owns the device, runs jobs in order, merges waiting recognitions.
fn gpu_thread(jobs: mpsc::Receiver<Job>, done: mpsc::Sender<Done>, threshold: f32, models: String, ready: mpsc::Sender<Result<String, String>>) {
    let setup = (|| -> Result<_, String> {
        let lines: usize = std::env::var("LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let gpu = gpu::Gpu::new(lines)?;
        let det = gpu::Detector::new(&gpu, &onnx::Onnx::load(&format!("{models}/text-detection.onnx"))?)?;
        let rec = gpu::Recognizer::new(&gpu, &onnx::Onnx::load(&format!("{models}/text-recognition.onnx"))?, lines)?;
        Ok((gpu, det, rec))
    })();
    let (gpu, det, rec) = match setup {
        Ok(s) => {
            let _ = ready.send(Ok(s.0.adapter.clone()));
            s
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    // MERGE=n: hold recognitions until n pages wait (or the last page came), so the GRU's
    // fixed 601 steps are paid once for several pages.
    let merge: usize = std::env::var("MERGE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut held: Vec<RecPage> = Vec::new();
    let mut pending: Vec<Job> = Vec::new();
    loop {
        if pending.is_empty() {
            match jobs.recv() {
                Ok(j) => pending.push(j),
                Err(_) => return,
            }
        }
        pending.extend(jobs.try_iter());
        // Detections first: the CPU waits for them; recognitions merge into one pass.
        let (detects, recs): (Vec<Job>, Vec<Job>) = pending.drain(..).partition(|j| matches!(j, Job::Detect { .. }));
        for j in detects {
            if let Job::Detect { page, rgb, w, h } = j {
                let _ = done.send(Done::Mask(page, det.detect_page(&gpu, &rgb, w, h, threshold)));
            }
        }
        held.extend(recs.into_iter().filter_map(|j| if let Job::Recognize(r) = j { Some(r) } else { None }));
        if held.is_empty() || (held.len() < merge && !held.iter().any(|r| r.last)) {
            continue;
        }
        let recs: Vec<RecPage> = std::mem::take(&mut held);
        let all: Vec<(Vec<f32>, usize, usize)> = recs.iter().flat_map(|r| r.batches.iter().cloned()).collect();
        if all.is_empty() {
            for r in recs {
                let _ = done.send(Done::Part(r.page, Ok(Vec::new())));
            }
            continue;
        }
        match rec.recognize(&gpu, &all) {
            Ok((logits, _, lens)) => {
                let ntot: usize = recs.iter().map(|r| r.order.len()).sum();
                let mut k0 = 0;
                for r in recs {
                    let part: Vec<(usize, String)> =
                        r.order.iter().enumerate().map(|(k, &i)| (i, decode(&logits, k0 + k, ntot, lens[k0 + k], 97, r.widths[k]))).collect();
                    k0 += r.order.len();
                    let _ = done.send(Done::Part(r.page, Ok(part)));
                }
            }
            Err(e) => {
                for r in recs {
                    let _ = done.send(Done::Part(r.page, Err(e.clone())));
                }
            }
        }
    }
}

/// ocrs' line images, in parallel, batched as ocrs batches them.
fn line_batches(engine: &OcrEngine, input: &ocrs::OcrInput, lines: &[Vec<RotatedRect>]) -> Result<(Vec<(Vec<f32>, usize, usize)>, Vec<usize>, Vec<usize>), String> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = lines.len().div_ceil(threads).max(1);
    let prepared: Vec<Result<(usize, NdTensor<f32, 2>), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = lines
            .chunks(chunk)
            .enumerate()
            .map(|(c, ls)| {
                scope.spawn(move || {
                    ls.iter()
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
        groups.entry(img.size(1).next_multiple_of(50)).or_default().push((i, img));
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
    Ok((batches, order, widths))
}

fn main() -> Result<(), String> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let models = std::env::var("MODELS").unwrap_or_else(|_| "models".into());
    let repeat: usize = std::env::var("REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(rten::Model::load_file(format!("{models}/text-detection.onnx")).map_err(|e| e.to_string())?),
        recognition_model: Some(rten::Model::load_file(format!("{models}/text-recognition.onnx")).map_err(|e| e.to_string())?),
        ..Default::default()
    })
    .map_err(|e| e.to_string())?;
    // Pages decoded up front for both paths (the app gets them from its renderer).
    let mut pages = Vec::new();
    for _ in 0..repeat {
        for p in &paths {
            let img = image::open(p).map_err(|e| e.to_string())?.into_rgb8();
            let (w, h) = img.dimensions();
            pages.push((std::sync::Arc::new(img.into_raw()), w as usize, h as usize));
        }
    }

    // --- ocrs on the CPU, page by page; warm-up on the first page.
    let ocr_page = |rgb: &[u8], w: usize, h: usize| -> Result<(Vec<String>, usize), String> {
        let input = engine.prepare_input(ImageSource::from_bytes(rgb, (w as u32, h as u32)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let words = engine.detect_words(&input).map_err(|e| e.to_string())?;
        let lines = engine.find_text_lines(&input, &words);
        let text = engine.recognize_text(&input, &lines).map_err(|e| e.to_string())?.into_iter().map(|l| l.map(|l| l.to_string()).unwrap_or_default()).collect();
        Ok((text, words.len()))
    };
    ocr_page(&pages[0].0, pages[0].1, pages[0].2)?;
    let s = Instant::now();
    let mut cpu = Vec::new();
    for (rgb, w, h) in &pages {
        cpu.push(ocr_page(rgb, *w, *h)?);
    }
    let cpu_ms = ms(s);

    // --- GPU thread + CPU, overlapped.
    let (job_tx, job_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let threshold = engine.detection_threshold();
    let (cpu_tx, cpu_rx) = mpsc::channel();
    let (cpu_done, cpu_models) = (done_tx.clone(), models.clone());
    let cpu_worker = std::thread::spawn(move || cpu_thread(cpu_rx, cpu_done, cpu_models));
    let cpu_share: f64 = std::env::var("CPUSHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let gpu = std::thread::spawn(move || gpu_thread(job_rx, done_tx, threshold, models, ready_tx));
    let adapter = ready_rx.recv().map_err(|e| e.to_string())??;
    println!("GPU: {adapter}");
    let run = |pages: &[(std::sync::Arc<Vec<u8>>, usize, usize)]| -> Result<(Vec<Vec<String>>, Vec<usize>, f64, f64), String> {
        let start = Instant::now();
        let mut masks: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut texts: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        let mut parts_left: BTreeMap<usize, usize> = BTreeMap::new();
        let mut finished = 0;
        let mut word_counts = vec![0; pages.len()];
        let mut cpu_busy = 0.0;
        let handle = |d: Done, masks: &mut BTreeMap<usize, Vec<u32>>, texts: &mut BTreeMap<usize, Vec<String>>, left: &mut BTreeMap<usize, usize>, finished: &mut usize| -> Result<(), String> {
            match d {
                Done::Mask(p, m) => {
                    masks.insert(p, m?);
                }
                Done::Part(p, t) => {
                    let page = texts.entry(p).or_default();
                    for (i, line) in t? {
                        if page.len() <= i {
                            page.resize(i + 1, String::new());
                        }
                        page[i] = line;
                    }
                    let l = left.entry(p).or_insert(2);
                    *l -= 1;
                    if *l == 0 {
                        *finished += 1;
                    }
                }
            }
            Ok(())
        };
        let send = |k: usize| {
            pages.get(k).map(|(rgb, w, h)| job_tx.send(Job::Detect { page: k, rgb: rgb.clone(), w: *w, h: *h }))
        };
        send(0);
        for (k, (rgb, w, h)) in pages.iter().enumerate() {
            let c = Instant::now();
            let input = engine.prepare_input(ImageSource::from_bytes(rgb, (*w as u32, *h as u32)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            cpu_busy += ms(c);
            while !masks.contains_key(&k) {
                let d = done_rx.recv().map_err(|e| e.to_string())?;
                handle(d, &mut masks, &mut texts, &mut parts_left, &mut finished)?;
            }
            // The next page's detection queues while this page's CPU work runs.
            send(k + 1);
            let c = Instant::now();
            let mask = masks.remove(&k).unwrap_or_default();
            let wr = word_rects(&mask, *w, *h);
            word_counts[k] = wr.len();
            let mut lines = engine.find_text_lines(&input, &wr);
            // EXPANDEND / EXPANDV: widen the first and last word of each found line (and all
            // words vertically), so the line image keeps the edge characters; line finding is
            // unaffected.
            let end: f32 = std::env::var("EXPANDEND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let vert: f32 = std::env::var("EXPANDV").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            if end > 0.0 || vert > 0.0 {
                for line in &mut lines {
                    let n = line.len();
                    for (i, r) in line.iter_mut().enumerate() {
                        let grow = if i == 0 || i + 1 == n { end } else { 0.0 };
                        r.resize(r.width() + 2. * grow, r.height() + 2. * vert);
                    }
                }
            }
            // RELV / RELEND: the same as fractions of the line's median word height, so they
            // scale with the resolution. A rect whose up axis is horizontal has width and
            // height swapped.
            let relv: f32 = std::env::var("RELV").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15);
            let relend: f32 = std::env::var("RELEND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1);
            if relv > 0.0 || relend > 0.0 {
                for line in &mut lines {
                    let upright = |r: &RotatedRect| r.up_axis().y.abs() >= r.up_axis().x.abs();
                    let mut heights: Vec<f32> = line.iter().map(|r| if upright(r) { r.height() } else { r.width() }).collect();
                    heights.sort_by(f32::total_cmp);
                    let h = heights.get(heights.len() / 2).copied().unwrap_or(0.0);
                    let n = line.len();
                    for (i, r) in line.iter_mut().enumerate() {
                        let (along, across) = (if i == 0 || i + 1 == n { relend * h } else { 0.0 }, relv * h);
                        if upright(r) {
                            r.resize(r.width() + 2. * along, r.height() + 2. * across);
                        } else {
                            r.resize(r.width() + 2. * across, r.height() + 2. * along);
                        }
                    }
                }
            }
            let (batches, order, widths) = line_batches(&engine, &input, &lines)?;
            cpu_busy += ms(c);
            // The last batches, up to CPUSHARE of the work (lines x width), go to rten on the CPU.
            let cost: Vec<usize> = batches.iter().map(|b| b.1 * b.2).collect();
            let total: usize = cost.iter().sum();
            let mut split = batches.len();
            let mut taken = 0;
            while split > 0 && ((taken + cost[split - 1]) as f64) <= cpu_share * total as f64 {
                split -= 1;
                taken += cost[split];
            }
            let lines_before: usize = batches[..split].iter().map(|b| b.1).sum();
            let mut batches = batches;
            let cpu_batches = batches.split_off(split);
            texts.insert(k, vec![String::new(); lines.len()]);
            cpu_tx
                .send(RecPage { page: k, batches: cpu_batches, order: order[lines_before..].to_vec(), widths: widths[lines_before..].to_vec(), lines: lines.len(), last: false })
                .map_err(|e| e.to_string())?;
            let page = RecPage { page: k, batches, order: order[..lines_before].to_vec(), widths: widths[..lines_before].to_vec(), lines: lines.len(), last: k + 1 == pages.len() };
            job_tx.send(Job::Recognize(page)).map_err(|e| e.to_string())?;
        }
        while finished < pages.len() {
            let d = done_rx.recv().map_err(|e| e.to_string())?;
            handle(d, &mut masks, &mut texts, &mut parts_left, &mut finished)?;
        }
        Ok((texts.into_values().collect(), word_counts, ms(start), cpu_busy))
    };
    run(&pages[..1])?; // warm-up
    let (gpu_text, gpu_words, gpu_ms, cpu_busy) = run(&pages)?;
    drop(job_tx);
    drop(cpu_tx);
    let _ = gpu.join();
    let _ = cpu_worker.join();

    // DUMP=dir: the text of each page from both paths, for comparing with other engines.
    if let Ok(dir) = std::env::var("DUMP") {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        for (k, ((ct, _), gt)) in cpu.iter().zip(&gpu_text).enumerate().take(paths.len()) {
            std::fs::write(format!("{dir}/ocrs_{k}.txt"), ct.join("\n")).map_err(|e| e.to_string())?;
            std::fs::write(format!("{dir}/gpu_{k}.txt"), gt.join("\n")).map_err(|e| e.to_string())?;
        }
    }
    let (mut same, mut total, mut words_same) = (0, 0, 0);
    for (k, ((ct, cw), gt)) in cpu.iter().zip(&gpu_text).enumerate() {
        words_same += usize::from(*cw == gpu_words[k]);
        for (i, (a, b)) in gt.iter().zip(ct).enumerate() {
            total += 1;
            if words(a) == words(b) {
                same += 1;
            } else if total - same <= 3 {
                println!("  page {k} line {i}:\n    cpu: {b}\n    gpu: {a}");
            }
        }
        if gt.len() != ct.len() {
            println!("  page {k}: {} lines on the GPU path, {} with ocrs", gt.len(), ct.len());
        }
    }
    println!("{} pages: same word count {words_same}/{}, same text {same}/{total} lines", pages.len(), pages.len());
    println!("ocrs on the CPU, page by page:   {cpu_ms:.0} ms ({:.0} ms a page)", cpu_ms / pages.len() as f64);
    println!("GPU + CPU overlapped:            {gpu_ms:.0} ms ({:.0} ms a page), CPU busy {cpu_busy:.0} ms   ({:.1}x)", gpu_ms / pages.len() as f64, cpu_ms / gpu_ms);
    Ok(())
}
