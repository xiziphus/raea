//! `raea motion "LINE" [--keys KEYS.json] ...`: body motion made on this machine by Kimodo (raea-sys-kimodo): a SOMA clip
//! from a line of words and, with --keys, Kimodo constraints (full-body keyframes, hand or foot targets, a root path:
//! what sdk/body/tools/between.ts writes). The clip is the file sdk/body/tools/fromkimodo.ts reads, the same one the
//! cloud runner wrote (docs/character.feature CH15b, CH15c).
//!
//! Which backend makes it is the machine's: MLX on a Mac (Apple silicon), PyTorch elsewhere (Linux, Windows), in the
//! environment `raea setup motion` makes (raea-sys-kimodo torch.rs). Both hand decode.rs the sampler's features, so
//! both write one file; their noise differs, so a seed is a different take on each.

use std::path::PathBuf;

use raea_runtime_bundle::flag;

type R<T> = Result<T, String>;

pub const USAGE: &str = "raea motion \"LINE\" [--keys KEYS.json] [--seconds 5 | --frames N] [--seed 1] [--steps 100] [-o CLIP.json]";
pub const ABOUT: &str = "body motion made on this machine by Kimodo (NVIDIA): a SOMA clip from a line of words and, with --keys, Kimodo constraints (full-body keyframes, hand or foot targets, a root path, as sdk/body/tools/between.ts writes them), as the clip sdk/body/tools/fromkimodo.ts reads. On a Mac it runs on MLX (Apple silicon's GPU): the first use fetches the motion model (1.1 GB) into raea's models folder, and the first new line its text encoder (8.8 GB: Llama 3 8B in 8 bits with LLM2Vec's adapters). On Linux and Windows it runs upstream Kimodo on PyTorch, set up once by raea setup motion: on an NVIDIA GPU with CUDA (the text encoder, 16 GB in bf16, on the CPU unless the card has 18 GB or more), else on the CPU alone; reading a new line on the CPU takes about 18 GB of memory. A line's words are read once and kept. Licences: Kimodo's weights NVIDIA's Open Model License, Meta Llama 3 the Meta Llama 3 Community License, LLM2Vec's adapters MIT, Kimodo's code Apache-2.0";

pub const SETUP_USAGE: &str = "raea setup motion [--check]";
pub const SETUP_ABOUT: &str = "what raea motion runs on, made where missing (--check only says): on a Mac the MLX models in raea's models folder (the motion model 1.1 GB, the text encoder 8.8 GB); on Linux and Windows, in raea's models folder's kimodo-torch, a Python 3.11 environment made by uv (which it needs on PATH: https://docs.astral.sh/uv) with PyTorch (its CUDA build where there is an NVIDIA driver, about 8 GB, else CPU only, about 2 GB) and upstream Kimodo (nv-tlabs/kimodo at e1c219c, Apache-2.0), then the weights: Kimodo-SOMA-RP-v1 (1.1 GB, NVIDIA Open Model License), LLM2Vec's two adapters (0.3 GB, MIT) and Meta Llama 3 8B Instruct in bf16 (16 GB, Meta Llama 3 Community License, from NousResearch's ungated copy, checked file by file against Meta's SHA-256); about 26 GB in all. It says what this machine has and where each network runs: the denoiser on CUDA when there is an NVIDIA GPU, else the CPU; the text encoder beside it on a GPU of 18 GB or more, else the CPU";

fn number(args: &[String], name: &str, default: f64) -> R<f64> {
    match flag(args, name) {
        None => Ok(default),
        Some(v) => v.parse::<f64>().ok().filter(|n| n.is_finite() && *n >= 0.0).ok_or(format!("{name} takes a number, got `{v}`")),
    }
}

fn line(args: &[String]) -> String {
    const WITH_VALUE: [&str; 6] = ["--keys", "--seconds", "--frames", "--seed", "--steps", "-o"];
    let mut out = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if WITH_VALUE.contains(&args[i].as_str()) {
            i += 2;
            continue;
        }
        if !args[i].starts_with("--") {
            out.push(args[i].as_str());
        }
        i += 1;
    }
    out.join(" ")
}

/// what a call asks for, read from its words and flags
struct Ask {
    words: String,
    keys: Option<String>,
    constraints: Option<serde_json::Value>,
    frames: usize,
    seed: u64,
    steps: usize,
    out: Option<String>,
}

pub fn motion(args: &[String]) -> R<String> {
    let words = line(args);
    let keys = flag(args, "--keys");
    if words.trim().is_empty() && keys.is_none() {
        return Err(format!("{USAGE}\n  ({ABOUT})"));
    }
    let frames = match flag(args, "--frames") {
        Some(_) => number(args, "--frames", 150.0)? as usize,
        None => (number(args, "--seconds", 5.0)? * 30.0).round() as usize,
    };
    if !(2..=600).contains(&frames) {
        return Err(format!("a clip is 2 to 600 frames (20 s at 30 a second), asked {frames}"));
    }
    let constraints = match &keys {
        Some(p) => Some(serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?),
        None => None,
    };
    let (seed, steps) = (number(args, "--seed", 1.0)? as u64, (number(args, "--steps", 100.0)? as usize).max(2));
    make(Ask { words, keys, constraints, frames, seed, steps, out: flag(args, "-o") })
}

/// -o, else a file named by the line's words in the working folder
fn out_path(out: Option<String>, clean: &str) -> PathBuf {
    PathBuf::from(out.unwrap_or_else(|| {
        let slug = clean.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect::<Vec<_>>().join("-");
        format!("{}.clip.json", if slug.is_empty() { "keys".to_string() } else { slug.chars().take(60).collect() })
    }))
}

/// how far the hips end from where they start, on the ground
fn travel(m: &raea_sys_kimodo::decode::Motion) -> f32 {
    let r = &m.root;
    let (a, b) = (r[0], r[r.len() - 1]);
    ((b[0] - a[0]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

/// Kimodo's PyTorch environment: raea's models folder's kimodo-torch
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn torch_env() -> raea_sys_kimodo::torch::Torch {
    raea_sys_kimodo::torch::Torch::at(raea_io_home::models().join("kimodo-torch"))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn make(ask: Ask) -> R<String> {
    use raea_sys_kimodo::{decode, skeleton::Skeleton, stats::Stats, torch};
    let t = torch_env();
    let missing = t.missing();
    if !missing.is_empty() {
        return Err(format!(
            "raea motion runs Kimodo on PyTorch on this machine (MLX is Apple silicon's), and its {} isn't set up in {} yet: raea setup motion makes it (uv, then about 26 GB: PyTorch, Kimodo's code and weights, Llama 3 8B Instruct for the text encoder)",
            missing.join(", the "),
            t.dir.display()
        ));
    }
    let keys = ask.keys.as_ref().map(std::path::Path::new);
    let job = torch::Job { line: &ask.words, frames: ask.frames, seed: ask.seed, steps: ask.steps, keys, cfg: (2.0, 2.0) };
    let made = t.make(&job).map_err(|e| format!("raea motion: {e}"))?;
    let (stats, sk) = (Stats::load(&t.checkpoint).map_err(|e| format!("raea motion: {e}"))?, Skeleton::soma().map_err(|e| format!("raea motion: {e}"))?);
    let motion = decode::decode(&made.features, &stats, &sk);
    let info = &made.info;
    let clean = info["line"].as_str().unwrap_or(&ask.words).to_string();
    let (dev, text_dev) = (info["motion"].as_str().unwrap_or("?"), info["text"].as_str().unwrap_or("?"));
    let on = match info["gpu"].as_str().filter(|_| dev == "cuda") {
        Some(g) => format!("on PyTorch {} with CUDA ({g}; the text encoder on {text_dev}), upstream nv-tlabs/kimodo {}", info["torch"].as_str().unwrap_or("?"), &torch::KIMODO[..7]),
        None => format!("on PyTorch {} on the CPU, upstream nv-tlabs/kimodo {}", info["torch"].as_str().unwrap_or("?"), &torch::KIMODO[..7]),
    };
    let clip = decode::clip_json_on(&motion, &sk, ask.seed, ask.steps, &clean, ask.constraints.is_some(), &on);
    let out = out_path(ask.out, &clean);
    std::fs::write(&out, serde_json::to_string(&clip).unwrap_or_default()).map_err(|e| format!("{}: {e}", out.display()))?;
    // the hips as PyTorch decoded them against decode.rs's: the features were read as Kimodo wrote them
    let ours: Vec<[f32; 3]> = motion.joints77.iter().map(|f| f[0]).collect();
    let gap = torch::root_gap(&info["root"], &ours).map(|g| format!(", decoded here within {:.3} mm of PyTorch's own", g * 1000.0)).unwrap_or_default();
    let f = |k: &str| info[k].as_f64().unwrap_or(0.0);
    let read = if f("read_s") > 0.0 { format!(", reading the line {:.1} s after {:.1} s loading its encoder", f("read_s"), f("text_load_s")) } else { String::new() };
    Ok(format!(
        "APPLIED  motion {:?}{}, {} frames ({:.1} s), seed {}, {} steps -> {} (made {on} in {:.1} s after {:.1} s loading{read}; the hips end {:.2} m from where they start{gap})",
        clean,
        if ask.keys.is_some() { " with keys" } else { "" },
        ask.frames,
        ask.frames as f64 / 30.0,
        ask.seed,
        ask.steps,
        out.display(),
        f("gen_s"),
        f("load_s"),
        travel(&motion),
    ))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn make(ask: Ask) -> R<String> {
    // float32 as checked against PyTorch: MLX's TF32 matmuls off, before the first one
    if std::env::var_os("MLX_ENABLE_TF32").is_none() {
        unsafe { std::env::set_var("MLX_ENABLE_TF32", "0") };
    }
    let t0 = std::time::Instant::now();
    let mut progress = fetching();
    let mut k = raea_sys_kimodo::Kimodo::fetch(&mut progress).map_err(|e| format!("raea motion: {e}"))?;
    let (clean, emb) = k.encode(&ask.words).map_err(|e| format!("raea motion: {e}"))?;
    let load_s = t0.elapsed().as_secs_f64();
    let made_from = raea_sys_kimodo::Ask { line: &ask.words, frames: ask.frames, seed: ask.seed, steps: ask.steps, constraints: ask.constraints.as_ref(), cfg: (2.0, 2.0) };
    let t1 = std::time::Instant::now();
    let made = k.make(&made_from, emb.as_deref(), None).map_err(|e| format!("raea motion: {e}"))?;
    let gen_s = t1.elapsed().as_secs_f64();
    let clip = raea_sys_kimodo::decode::clip_json(&made.motion, &k.skeleton, ask.seed, ask.steps, &clean, ask.constraints.is_some());
    let out = out_path(ask.out, &clean);
    std::fs::write(&out, serde_json::to_string(&clip).unwrap_or_default()).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(format!(
        "APPLIED  motion {:?}{}, {} frames ({:.1} s), seed {}, {} steps -> {} (made on this Mac in {gen_s:.1} s after {load_s:.1} s loading and reading the line; the hips end {:.2} m from where they start)",
        clean,
        if ask.keys.is_some() { " with keys" } else { "" },
        ask.frames,
        ask.frames as f64 / 30.0,
        ask.seed,
        ask.steps,
        out.display(),
        travel(&made.motion)
    ))
}

/// a line every 512 MB of each file fetched
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fetching() -> impl FnMut(&str, u64) {
    let mut shown = std::collections::HashMap::<String, u64>::new();
    move |file: &str, n: u64| {
        if shown.insert(file.to_string(), n >> 29) != Some(n >> 29) {
            eprintln!("raea motion: fetching {file}: {} MB", n >> 20);
        }
    }
}

/// What a folder takes on disk, through a link (the Mac's model folders can be links to their files).
fn disk(dir: &std::path::Path) -> String {
    raea_io_home::disk_use(dir, true)
}

/// `raea setup motion [--check]`: what raea motion runs on here, made where missing (--check only says), and where it is.
pub fn setup(args: &[String]) -> R<String> {
    let check = args.iter().any(|a| a == "--check");
    if let Some(a) = args.iter().find(|a| *a != "--check") {
        return Err(format!("{SETUP_USAGE}: no `{a}`"));
    }
    setup_here(check)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn setup_here(check: bool) -> R<String> {
    let was = raea_sys_kimodo::installed();
    if !check && was.iter().any(|m| !m.2) {
        raea_io_home::offline("raea setup motion's models").map_err(|e| format!("raea setup motion: {e}"))?;
        let mut progress = fetching();
        raea_sys_kimodo::fetch_all(&mut progress).map_err(|e| format!("raea setup motion: {e}"))?;
    }
    let now = raea_sys_kimodo::installed();
    let mut lines = vec!["  backend      MLX, on this Mac's GPU (raea-sys-kimodo, no Python)".to_string()];
    for ((name, dir, had), (_, _, has)) in was.iter().zip(&now) {
        let what = if *name == "kimodo" { "the motion model: Kimodo-SOMA-RP-v1, 1.1 GB, NVIDIA Open Model License" } else { "the text encoder: Llama 3 8B Instruct in 8 bits (Meta Llama 3 Community License) with LLM2Vec's adapters (MIT), 8.8 GB" };
        let state = match (had, has) {
            (true, _) => "installed",
            (false, true) => "installed now",
            (false, false) => "not installed: raea setup motion fetches it (raea motion does too, the first time it needs it)",
        };
        lines.push(format!("  {name:<12} {state}; {what}\n               {} ({})", dir.display(), if *has { disk(dir) } else { "-".into() }));
    }
    let ready = now.iter().all(|m| m.2);
    let head = match (check, ready) {
        (true, true) => "CHECKED  raea motion's models",
        (true, false) => "CHECKED  raea motion's models (some missing)",
        _ => "APPLIED  raea motion's models ready",
    };
    Ok(format!("{head}\n{}", lines.join("\n")))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn setup_here(check: bool) -> R<String> {
    let t = torch_env();
    let was = t.missing();
    let mut made = Vec::new();
    if !check && !was.is_empty() {
        raea_io_home::offline("raea setup motion's environment and weights").map_err(|e| format!("raea setup motion: {e}"))?;
        made = t.setup(&mut |s| eprintln!("raea setup motion: {s}")).map_err(|e| format!("raea setup motion: {e}"))?;
    }
    let now = t.missing();
    let mut lines: Vec<String> = made.iter().map(|m| format!("  {m}")).collect();
    lines.push(format!("  folder       {} ({})", t.dir.display(), if t.dir.exists() { disk(&t.dir) } else { "-".into() }));
    if now.is_empty() {
        match t.probe() {
            Ok(p) => {
                let gb = |k: &str| p[k].as_f64().map(|v| format!("{v:.0} GB")).unwrap_or_else(|| "?".into());
                let gpu = match p["gpu"].as_str() {
                    Some(g) => format!("{g}, {} (CUDA {})", gb("vram_gb"), p["cuda"].as_str().unwrap_or("?")),
                    None => "none that PyTorch sees (no NVIDIA GPU, or its driver)".into(),
                };
                lines.push(format!("  backend      PyTorch {}, upstream Kimodo {}", p["torch"].as_str().unwrap_or("?"), &raea_sys_kimodo::torch::KIMODO[..7]));
                lines.push(format!("  machine      GPU: {gpu}; memory {}; {} CPU threads", gb("ram_gb"), p["threads"]));
                lines.push(format!("  runs on      the denoiser: {}; the text encoder (a new line only): {}", p["motion"].as_str().unwrap_or("?"), p["text"].as_str().unwrap_or("?")));
                if p["text"] == "cpu" && p["ram_gb"].as_f64().is_some_and(|r| r < 18.0) {
                    lines.push(format!("  warning      a new line's encoder needs about 18 GB of memory on the CPU, and this machine has {}: lines already read still work", gb("ram_gb")));
                }
            }
            Err(e) => lines.push(format!("  backend      the environment didn't answer: {e}")),
        }
    } else {
        lines.push(format!("  missing      the {}: raea setup motion makes it (needs uv: https://docs.astral.sh/uv)", now.join(", the ")));
    }
    let head = match (check, now.is_empty()) {
        (true, true) => "CHECKED  raea motion's PyTorch environment",
        (true, false) => "CHECKED  raea motion's PyTorch environment (not set up)",
        _ => "APPLIED  raea motion's PyTorch environment ready",
    };
    Ok(format!("{head}\n{}", lines.join("\n")))
}
