//! Kimodo on this Mac (docs/character.feature CH15c): SOMA body motion from a line of text and constraints, made by
//! Kimodo-SOMA-RP-v1 (nv-tlabs/kimodo: code Apache-2.0; weights under NVIDIA's Open Model License, commercial use
//! allowed) on raea's MLX facility, with its text encoder (LLM2Vec, McGill-NLP, MIT adapters, on Llama 3 8B Instruct
//! in 8 bits: Meta's Llama 3 Community License) on MLX too. No GPU in the cloud and no Python.
//!
//! The networks are plain functions over the checkpoints' own safetensors (net.rs, text.rs), checked against the
//! PyTorch reference: the denoiser to 1e-4 on its features, a whole 100-step generation to hundredths of a millimetre
//! on the joints (tests.rs). Kimodo's post-processing (foot-skate cleanup and constraint enforcement, a C++ extension)
//! is not here: clips are post_processing=False, and raea's own foot planting runs on them downstream.
//!
//! The skeleton, the constraints, features to joints and the clip file are plain Rust and build everywhere; the
//! networks only on macOS. Elsewhere (Linux, Windows) torch.rs runs upstream Kimodo on PyTorch in an environment raea
//! makes, and decode.rs turns its features into the same clip.

pub mod constraints;
pub mod decode;
pub mod skeleton;
pub mod stats;
pub mod torch;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod net;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod sample;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod text;

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use model::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod model {
    use std::path::{Path, PathBuf};

    use anyhow::Result;
    use raea_sys_ml::models::{Spec, ensure};

    use crate::constraints::{Conditions, conditions};
    use crate::decode::{DIM, Motion, decode};
    use crate::net::Net;
    use crate::skeleton::Skeleton;
    use crate::stats::Stats;

    /// The motion model: 1.13 GB.
    pub const MOTION: Spec = Spec {
        name: "kimodo",
        files: &[
            ("nvidia/Kimodo-SOMA-RP-v1", "model.safetensors", "model.safetensors"),
            ("nvidia/Kimodo-SOMA-RP-v1", "stats/motion/global_root/mean.npy", "stats/motion/global_root/mean.npy"),
            ("nvidia/Kimodo-SOMA-RP-v1", "stats/motion/global_root/std.npy", "stats/motion/global_root/std.npy"),
            ("nvidia/Kimodo-SOMA-RP-v1", "stats/motion/local_root/mean.npy", "stats/motion/local_root/mean.npy"),
            ("nvidia/Kimodo-SOMA-RP-v1", "stats/motion/local_root/std.npy", "stats/motion/local_root/std.npy"),
            ("nvidia/Kimodo-SOMA-RP-v1", "stats/motion/body/mean.npy", "stats/motion/body/mean.npy"),
            ("nvidia/Kimodo-SOMA-RP-v1", "stats/motion/body/std.npy", "stats/motion/body/std.npy"),
            ("nvidia/Kimodo-SOMA-RP-v1", "LICENSE", "LICENSE"),
        ],
    };

    /// The text encoder: 8.5 GB of 8-bit Llama 3 and 336 MB of adapters. Fetched only when a line has words that
    /// haven't been read before.
    pub const TEXT: Spec = Spec {
        name: "kimodo-text",
        files: &[
            ("mlx-community/Meta-Llama-3-8B-Instruct-8bit", "config.json", "config.json"),
            ("mlx-community/Meta-Llama-3-8B-Instruct-8bit", "model-00001-of-00002.safetensors", "model-00001-of-00002.safetensors"),
            ("mlx-community/Meta-Llama-3-8B-Instruct-8bit", "model-00002-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "adapter_model.safetensors", "llm2vec/mntp.safetensors"),
            ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "adapter_model.safetensors", "llm2vec/supervised.safetensors"),
            ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "tokenizer.json", "llm2vec/tokenizer.json"),
        ],
    };

    /// Each model's folder in the models folder, and whether every file it needs is there (`raea setup motion --check`).
    pub fn installed() -> [(&'static str, PathBuf, bool); 2] {
        [&MOTION, &TEXT].map(|s| {
            let d = raea_sys_ml::models::root().join(s.name);
            let all = s.files.iter().all(|f| d.join(f.2).exists());
            (s.name, d, all)
        })
    }

    /// Both models fetched where missing (`raea setup motion`): the motion model, then the text encoder.
    pub fn fetch_all(progress: &mut dyn FnMut(&str, u64)) -> Result<[PathBuf; 2]> {
        Ok([ensure(&MOTION, progress)?, ensure(&TEXT, progress)?])
    }

    /// What a clip is made from.
    pub struct Ask<'a> {
        pub line: &'a str,
        pub frames: usize,
        pub seed: u64,
        pub steps: usize,
        /// a Kimodo constraints file's list (keyframes, end effectors, a root path), or none
        pub constraints: Option<&'a serde_json::Value>,
        /// guidance for the text and for the constraints (Kimodo's default 2 and 2)
        pub cfg: (f32, f32),
    }

    pub struct Made {
        pub motion: Motion,
        pub features: Vec<f32>,
        /// the line as the encoder read it (sanitised)
        pub line: String,
    }

    pub struct Kimodo {
        pub net: Net,
        pub stats: Stats,
        pub skeleton: Skeleton,
        text_dir: Option<PathBuf>,
        text: Option<crate::text::Encoder>,
        /// where each line's numbers are kept once read (CH15c: the encoder runs once a line), and whether the
        /// encoder may be fetched into the models folder when a line is new
        lines: Option<PathBuf>,
        fetch_text: bool,
    }

    /// what a line's numbers are kept under: FNV-1a over the encoder's name and the line as it reads it
    fn line_key(clean: &str) -> String {
        let h = "llm2vec-llama3-8b-instruct-q8\n".bytes().chain(clean.bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        format!("{h:016x}.f32")
    }

    impl Kimodo {
        /// From folders already on disk: the motion checkpoint, and the text encoder's (read on the first line with words).
        pub fn open(motion_dir: &Path, text_dir: Option<&Path>) -> Result<Self> {
            let stats = Stats::load(motion_dir)?;
            Ok(Kimodo { net: Net::load(motion_dir, &stats)?, stats, skeleton: Skeleton::soma()?, text_dir: text_dir.map(Path::to_path_buf), text: None, lines: None, fetch_text: false })
        }

        /// From raea's models folder, fetching the motion model if it's missing; the text encoder is fetched by the
        /// first line it hasn't read, and each line's numbers are kept beside the motion model.
        pub fn fetch(progress: &mut dyn FnMut(&str, u64)) -> Result<Self> {
            let m = ensure(&MOTION, progress)?;
            let mut k = Self::open(&m, None)?;
            k.lines = Some(m.join("lines"));
            k.fetch_text = true;
            Ok(k)
        }

        /// A line's 4096 numbers, or none for an empty line.
        pub fn encode(&mut self, line: &str) -> Result<(String, Option<Vec<f32>>)> {
            let clean = crate::text::sanitize(line);
            if clean.is_empty() {
                return Ok((clean, None));
            }
            let kept = self.lines.as_ref().map(|d| d.join(line_key(&clean)));
            if let Some(b) = kept.as_ref().and_then(|p| std::fs::read(p).ok()).filter(|b| b.len() == 4 * crate::net::TEXT_DIM as usize) {
                return Ok((clean, Some(b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())));
            }
            if self.text.is_none() {
                let dir = match (&self.text_dir, self.fetch_text) {
                    (Some(d), _) => d.clone(),
                    (None, true) => ensure(&TEXT, &mut |f, n| if n % (1 << 30) < (1 << 16) { eprintln!("kimodo: fetching the text encoder's {f}: {} MB", n >> 20) })?,
                    (None, false) => anyhow::bail!("kimodo: a line with words needs the text encoder's folder"),
                };
                self.text = Some(crate::text::Encoder::load(&dir)?);
            }
            let (t, e) = self.text.as_ref().unwrap().encode(line)?;
            if let Some(p) = kept {
                let b: Vec<u8> = e.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::create_dir_all(p.parent().unwrap()).and_then(|_| std::fs::write(&p, b));
            }
            Ok((t, Some(e)))
        }

        pub fn conditions(&self, c: &serde_json::Value, frames: usize) -> Result<Conditions> {
            conditions(c, frames, &self.stats, &self.skeleton)
        }

        /// One clip. `text` is the line's embedding (from `encode`), or none.
        pub fn make(&self, ask: &Ask, text: Option<&[f32]>, noise: Option<&[f32]>) -> Result<Made> {
            let cond = ask.constraints.map(|c| self.conditions(c, ask.frames)).transpose()?;
            let own;
            let noise = match noise {
                Some(n) => n,
                None => {
                    own = crate::sample::noise(ask.seed, ask.frames * DIM);
                    &own
                }
            };
            let features = crate::sample::sample(&self.net, noise, ask.frames, ask.steps, text, cond.as_ref().map(|c| (c.obs.as_slice(), c.mask.as_slice())), ask.cfg)?;
            Ok(Made { motion: decode(&features, &self.stats, &self.skeleton), features, line: crate::text::sanitize(ask.line) })
        }
    }
}
