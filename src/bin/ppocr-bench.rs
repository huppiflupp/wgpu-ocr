//! ocrs' detection and line finding (with line padding), then PP-OCRv6 recognition: on rten (CPU)
//! as the reference, and on the GPU once the executor runs it. Prints time per stage and writes
//! the text of each page (DUMP=dir) for scoring with bench/bench.py.

use std::collections::BTreeMap;
use std::time::Instant;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;
use wgpu_ocr::{gpu, nd, onnx, ppocr};

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// Line padding as in the ocrs pipeline (0.15 of the median word height above and below,
/// 0.1 before the first and after the last word); returns the padded line's bounding box.
fn padded_box(line: &[RotatedRect]) -> Option<[f32; 4]> {
    let upright = |r: &RotatedRect| r.up_axis().y.abs() >= r.up_axis().x.abs();
    let mut heights: Vec<f32> = line.iter().map(|r| if upright(r) { r.height() } else { r.width() }).collect();
    heights.sort_by(f32::total_cmp);
    let h = *heights.get(heights.len() / 2)?;
    let n = line.len();
    let padded: Vec<RotatedRect> = line
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut r = *r;
            let along = if i == 0 || i + 1 == n { 0.1 * h } else { 0.0 };
            if upright(&r) {
                r.resize(r.width() + 2. * along, r.height() + 2. * 0.15 * h);
            } else {
                r.resize(r.width() + 2. * 0.15 * h, r.height() + 2. * along);
            }
            r
        })
        .collect();
    let b = rten_imageproc::bounding_rect(padded.iter())?;
    Some([b.left(), b.top(), b.right(), b.bottom()])
}

fn main() -> Result<(), String> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let models = std::env::var("MODELS").unwrap_or_else(|_| "models".into());
    let rec_path = format!("{models}/ppocr/PP-OCRv6_small_rec.onnx");
    let dict = ppocr::Dictionary::load(&format!("{models}/ppocr/ppocrv6_dict.txt"))?;
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(rten::Model::load_file(format!("{models}/text-detection.onnx")).map_err(|e| e.to_string())?),
        recognition_model: Some(rten::Model::load_file(format!("{models}/text-recognition.onnx")).map_err(|e| e.to_string())?),
        ..Default::default()
    })
    .map_err(|e| e.to_string())?;
    // ENGINE: rten (CPU), gpu, or both (each batch on both, compared).
    let engine_kind = std::env::var("ENGINE").unwrap_or_else(|_| "gpu".into());
    let (use_rten, use_gpu) = (engine_kind != "gpu", engine_kind != "rten");
    let s = Instant::now();
    let rec = rten::Model::load_file(&rec_path).map_err(|e| e.to_string())?;
    println!("PP-OCRv6 rec loaded on rten in {:.0} ms, {} classes", ms(s), dict.classes());
    let s = Instant::now();
    let gpu = gpu::Gpu::new(8)?;
    let graph = nd::NdGraph::new(&gpu, &onnx::Onnx::load(&rec_path)?, engine_kind != "both")?;
    println!("PP-OCRv6 rec on the GPU ({}) in {:.0} ms", gpu.adapter, ms(s));
    let (mut max_diff, mut steps_same, mut steps_total) = (0f32, 0usize, 0usize);

    let (mut t_det, mut t_prep, mut t_rec, mut lines_total) = (0.0, 0.0, 0.0, 0);
    for (k, path) in paths.iter().enumerate() {
        let img = image::open(path).map_err(|e| e.to_string())?.into_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let s = Instant::now();
        let input = engine.prepare_input(ImageSource::from_bytes(img.as_raw(), (w as u32, h as u32)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let words = engine.detect_words(&input).map_err(|e| e.to_string())?;
        let lines = engine.find_text_lines(&input, &words);
        t_det += ms(s);
        let s = Instant::now();
        let crops: Vec<(Vec<f32>, usize)> = lines.iter().filter_map(|l| padded_box(l)).map(|b| ppocr::line_input(img.as_raw(), w, h, b)).collect();
        // Batches as PaddleOCR-style groups: width rounded up to 32, at most 16 lines.
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, (_, lw)) in crops.iter().enumerate() {
            groups.entry(lw.next_multiple_of(32)).or_default().push(i);
        }
        t_prep += ms(s);
        let s = Instant::now();
        let mut text = vec![String::new(); crops.len()];
        for (width, members) in &groups {
            let per_batch: usize = std::env::var("PPBATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
            for chunk in members.chunks(per_batch) {
                let refs: Vec<&(Vec<f32>, usize)> = chunk.iter().map(|&i| &crops[i]).collect();
                let data = ppocr::batch(&refs, *width);
                let shape = [chunk.len(), 3, ppocr::HEIGHT, *width];
                // Per line, the best class of each step, from rten and/or the GPU.
                let mut best_rten: Option<(Vec<u32>, usize, Vec<f32>)> = None;
                if use_rten {
                    let x = NdTensor::from_data(shape, data.clone());
                    let out: NdTensor<f32, 3> = rec.run_one(x.view().into(), None).map_err(|e| e.to_string())?.try_into().map_err(|e| format!("{e:?}"))?;
                    let [_, steps, classes] = out.shape();
                    let flat = out.to_vec();
                    let best = flat.chunks(classes).map(|row| (0..classes).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap_or(0) as u32).collect();
                    best_rten = Some((best, steps, flat));
                }
                let mut best_gpu: Option<(Vec<u32>, usize)> = None;
                if use_gpu {
                    let mut enc = gpu.encoder();
                    let mut scratch = Vec::new();
                    let out = graph.run(&gpu, &mut enc, nd::Tensor { buf: gpu.upload(&data), shape: shape.to_vec() }, &mut scratch)?;
                    let steps = out.shape[1];
                    if let Some((_, _, ref probs)) = best_rten {
                        // Validation: the GPU ran the whole graph, Softmax included.
                        let classes = out.shape[2];
                        let got = gpu.read(enc, &out.buf, probs.len());
                        for (a, b) in got.iter().zip(probs) {
                            max_diff = max_diff.max((a - b).abs());
                        }
                        let best = got.chunks(classes).map(|row| (0..classes).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap_or(0) as u32).collect();
                        best_gpu = Some((best, steps));
                    } else {
                        best_gpu = Some((graph.argmax(&gpu, enc, &out), steps));
                    }
                    gpu.recycle(scratch);
                }
                if let (Some((r, _, _)), Some((g, _))) = (&best_rten, &best_gpu) {
                    steps_total += r.len();
                    steps_same += r.iter().zip(g).filter(|(a, b)| a == b).count();
                }
                let (best, steps) = match (&best_gpu, &best_rten) {
                    (Some((b, s)), _) => (b.clone(), *s),
                    (None, Some((b, s, _))) => (b.clone(), *s),
                    _ => return Err("no engine".into()),
                };
                for (j, &i) in chunk.iter().enumerate() {
                    text[i] = dict.decode(&best[j * steps..(j + 1) * steps]);
                }
            }
        }
        t_rec += ms(s);
        lines_total += crops.len();
        if let Ok(dir) = std::env::var("DUMP") {
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            std::fs::write(format!("{dir}/ppocr-{engine_kind}_{k}.txt"), text.join("\n")).map_err(|e| e.to_string())?;
        }
        if k == 0 {
            for l in text.iter().take(3) {
                println!("  {l}");
            }
        }
    }
    let n = paths.len() as f64;
    if std::env::var_os("NDPROFILE").is_some() {
        for (k, t, c) in nd::profile().iter().take(16) {
            println!("  {k:<22} {c:>5} x {t:8.1} ms");
        }
    }
    if steps_total > 0 {
        println!("GPU vs rten: max |Δp| {max_diff:.6}, same best class at {steps_same}/{steps_total} steps");
    }
    println!(
        "{} pages, {lines_total} lines: detection + lines {:.0} ms/page, line images {:.0} ms/page, PP-OCRv6 ({engine_kind}) {:.0} ms/page",
        paths.len(),
        t_det / n,
        t_prep / n,
        t_rec / n
    );
    Ok(())
}
