//! PP-OCRv6 text-line recognition (PaddleOCR, Apache-2.0): its dictionary, its input
//! preprocessing and greedy CTC decoding. The model itself runs on rten (CPU) or on the GPU.

/// PaddleOCR's character list: blank (class 0), the dictionary in order, then the space.
pub struct Dictionary {
    chars: Vec<String>,
}

impl Dictionary {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut chars = vec![String::new()];
        chars.extend(text.lines().map(str::to_string));
        chars.push(" ".into());
        Ok(Self { chars })
    }

    pub fn classes(&self) -> usize {
        self.chars.len()
    }

    /// Greedy CTC over per-step class indices: repeats merged, blanks dropped.
    pub fn decode(&self, best: &[u32]) -> String {
        let mut out = String::new();
        let mut prev = 0u32;
        for &c in best {
            if c != 0 && c != prev {
                out.push_str(self.chars.get(c as usize).map_or("", String::as_str));
            }
            prev = c;
        }
        out
    }
}

/// The model's input height.
pub const HEIGHT: usize = 48;

/// One line cut from an RGB page (`rgb`, `w` x `h`) at the box `[left, top, right, bottom]`,
/// resized to height 48 keeping its aspect ratio, normalised as PaddleOCR does ((v / 255 - 0.5)
/// / 0.5, channels first). Returns the planar data and its width.
pub fn line_input(rgb: &[u8], w: usize, h: usize, rect: [f32; 4]) -> (Vec<f32>, usize) {
    let clampf = |v: f32, hi: usize| v.max(0.0).min(hi as f32) as u32;
    let (l, t, r, b) = (clampf(rect[0], w), clampf(rect[1], h), clampf(rect[2], w), clampf(rect[3], h));
    let (cw, ch) = (r.saturating_sub(l).max(1), b.saturating_sub(t).max(1));
    let mut crop = image::RgbImage::new(cw, ch);
    for y in 0..ch {
        let row = ((t + y) as usize * w + l as usize) * 3;
        if let (Some(src), Some(dst)) = (rgb.get(row..row + cw as usize * 3), crop.get_mut((y * cw * 3) as usize..((y + 1) * cw * 3) as usize)) {
            dst.copy_from_slice(src);
        }
    }
    let out_w = ((HEIGHT as f32 * cw as f32 / ch as f32).ceil() as usize).clamp(8, 3200);
    let resized = image::imageops::resize(&crop, out_w as u32, HEIGHT as u32, image::imageops::FilterType::Triangle);
    let mut data = vec![0f32; 3 * HEIGHT * out_w];
    for (x, y, p) in resized.enumerate_pixels() {
        for c in 0..3 {
            data[(c * HEIGHT + y as usize) * out_w + x as usize] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    (data, out_w)
}

/// Lines of possibly different widths as one [n, 3, 48, width] batch, padded with 0 on the
/// right (PaddleOCR pads its normalised batch with zeros).
pub fn batch(lines: &[&(Vec<f32>, usize)], width: usize) -> Vec<f32> {
    let mut out = vec![0f32; lines.len() * 3 * HEIGHT * width];
    for (k, (data, lw)) in lines.iter().enumerate() {
        for c in 0..3 {
            for y in 0..HEIGHT {
                let src = (c * HEIGHT + y) * lw;
                let dst = ((k * 3 + c) * HEIGHT + y) * width;
                out[dst..dst + lw].copy_from_slice(&data[src..src + lw]);
            }
        }
    }
    out
}
