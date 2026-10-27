//! PP-OCR text detection around the model: the input as PaddleOCR prepares it, and the boxes
//! from its probability map, by the DB (Differentiable Binarization) post-processing the model
//! was trained for. Parameters are the ones in the model's inference.yml.

use rten_imageproc::{BoundingRect, PointF, RetrievalMode, RotatedRect, find_contours, min_area_rect, simplify_polygon};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

/// The model input: [3, h, w], and the factors from it back to the page.
pub struct DetInput {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub to_page: [f32; 2],
    /// The input already on the GPU, when it was prepared there.
    pub tensor: Option<crate::nd::Tensor>,
}

impl DetInput {
    pub fn with_tensor(mut self, t: crate::nd::Tensor) -> Self {
        self.tensor = Some(t);
        self
    }
}

/// The page as the detector wants it: resized to multiples of 32 (at most `max_side` on the
/// long side, otherwise the page's own size), channels in BGR order, normalised with the
/// ImageNet mean and deviation.
pub fn det_input(rgb: &[u8], w: usize, h: usize, max_side: usize) -> DetInput {
    let ratio = if w.max(h) > max_side { max_side as f32 / w.max(h) as f32 } else { 1.0 };
    let round = |v: f32| ((v / 32.0).round() as usize * 32).max(32);
    let (rw, rh) = (round(w as f32 * ratio), round(h as f32 * ratio));
    let page = image::RgbImage::from_raw(w as u32, h as u32, rgb.to_vec()).unwrap_or_else(|| image::RgbImage::new(1, 1));
    let resized = image::imageops::resize(&page, rw as u32, rh as u32, image::imageops::FilterType::Triangle);
    let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
    let mut data = vec![0f32; 3 * rw * rh];
    for (x, y, p) in resized.enumerate_pixels() {
        // BGR: channel 0 is blue.
        for (c, v) in [p[2], p[1], p[0]].into_iter().enumerate() {
            data[(c * rh + y as usize) * rw + x as usize] = (v as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    DetInput { data, w: rw, h: rh, to_page: [w as f32 / rw as f32, h as f32 / rh as f32], tensor: None }
}

/// DB post-processing parameters.
#[derive(Clone, Copy)]
pub struct Db {
    pub thresh: f32,
    pub box_thresh: f32,
    pub unclip_ratio: f32,
    pub min_size: f32,
}

impl Default for Db {
    /// PP-OCRv6_small_det's inference.yml.
    fn default() -> Self {
        Self { thresh: 0.2, box_thresh: 0.45, unclip_ratio: 1.4, min_size: 3.0 }
    }
}

/// The mean probability inside `r`.
fn score(prob: &[f32], w: usize, h: usize, r: &RotatedRect) -> f32 {
    let b = r.bounding_rect();
    let (x0, y0) = (b.left().floor().max(0.0) as usize, b.top().floor().max(0.0) as usize);
    let (x1, y1) = ((b.right().ceil() as usize).min(w.saturating_sub(1)), (b.bottom().ceil() as usize).min(h.saturating_sub(1)));
    let (c, up) = (r.center(), r.up_axis());
    let (hw, hh) = (r.width() / 2.0, r.height() / 2.0);
    let (mut sum, mut n) = (0.0f32, 0usize);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (dx, dy) = (x as f32 - c.x, y as f32 - c.y);
            // Along the up axis is the height; across it the width.
            let along_up = dx * up.x + dy * up.y;
            let across = dx * -up.y + dy * up.x;
            if along_up.abs() <= hh && across.abs() <= hw {
                sum += prob[y * w + x];
                n += 1;
            }
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Text boxes from the detector's probability map (`w` x `h`), in page coordinates, in reading
/// order (top to bottom; left to right within about a line).
pub fn boxes(prob: &[f32], w: usize, h: usize, to_page: [f32; 2], db: Db) -> Vec<RotatedRect> {
    let mask = NdTensor::from_data([h, w], prob.iter().map(|&p| p > db.thresh).collect::<Vec<bool>>());
    let mut out = Vec::new();
    let contours = find_contours(mask.view(), RetrievalMode::External);
    if std::env::var_os("DETDEBUG").is_some() {
        let on = prob.iter().filter(|&&p| p > db.thresh).count();
        eprintln!("  db: {w}x{h}, {on} pixels over {} ({:.1} %), {} contours", db.thresh, 100.0 * on as f32 / prob.len() as f32, contours.len());
    }
    for poly in contours.iter() {
        let points: Vec<PointF> = poly.iter().map(|p| p.to_f32()).collect();
        // As ocrs does: simplify the contour first (min_area_rect expects few points).
        let hull = simplify_polygon(&points, 2.0);
        let Some(r) = min_area_rect(&hull) else { continue };
        if std::env::var_os("DETDEBUG").is_some() && out.len() < 5 {
            eprintln!("    contour {} points, rect {:.0}x{:.0} at ({:.0},{:.0}), score {:.2}", points.len(), r.width(), r.height(), r.center().x, r.center().y, score(prob, w, h, &r));
        }
        if r.width().min(r.height()) < db.min_size || score(prob, w, h, &r) < db.box_thresh {
            continue;
        }
        // Unclip: the box grown by area * ratio / perimeter on every side (the DB paper's offset).
        let (rw, rh) = (r.width(), r.height());
        let d = rw * rh * db.unclip_ratio / (2.0 * (rw + rh)).max(1e-6);
        let (nw, nh) = (rw + 2.0 * d, rh + 2.0 * d);
        if nw.min(nh) < db.min_size + 2.0 {
            continue;
        }
        let c = r.center();
        let s = (to_page[0] + to_page[1]) / 2.0;
        out.push(RotatedRect::new(PointF::from_yx(c.y * to_page[1], c.x * to_page[0]), r.up_axis(), nw * s, nh * s));
    }
    // Reading order: by the top edge, and boxes within 10 px of each other's top by the left edge.
    out.sort_by(|a, b| a.bounding_rect().top().total_cmp(&b.bounding_rect().top()));
    for i in 0..out.len() {
        let mut j = i;
        while j > 0 {
            let (a, b) = (out[j - 1].bounding_rect(), out[j].bounding_rect());
            if (b.top() - a.top()).abs() < 10.0 && b.left() < a.left() {
                out.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
    out
}
