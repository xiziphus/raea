# raea on Intel Macs (x86_64 macOS port)

Upstream raea ships only for Apple Silicon Macs. This branch builds and runs it from source on x86_64 macOS.

- Source: https://github.com/shinyobjectz/raea at 53f9cba (tag v0.1.10-rc3), ported 2026-11-25.
- Binary: `target/release/raea` (x86_64, JIT entitlement signed); link it as `~/.raea/bin/raea` and put `~/.raea/bin`
  on PATH. Write the tag into an untracked `VERSION` file at the repo root (`git describe --tags | sed 's/^v//' > VERSION`)
  so `raea --version` and the video-decompiler skill's `vd doctor` accept it (a checkout otherwise reports 0.1.0-checkout).
- Tested 2026-11-26 on an Intel Mac (macOS 15): Pong sample renders (4,668 frames, 1920x1080, 11.7 ms a frame on the GPU); `raea init` +
  `raea render` in a fresh folder; video-decompiler `vd doctor --full` passes (two films made and looked at).

## What was changed for Intel

1. `.cargo/config.toml`: `[target.x86_64-apple-darwin] linker = "tools/link-jit.sh"` (same JIT-entitled link as arm64).
2. ONNX Runtime loaded at run time on Intel Macs, as on Windows x64 (ort ships no x86_64 macOS build):
   the `cfg(all(windows, target_arch = "x86_64"))` dependency blocks in `systems/{song,speech,ml,index,columns}` and
   `apps/voice` now read `cfg(all(target_arch = "x86_64", any(windows, target_os = "macos")))`.
3. Kimodo motion (`systems/kimodo`, `apps/motion`): its MLX path is Apple-Silicon only, so `target_os = "macos"` now
   reads `all(target_os = "macos", target_arch = "aarch64")`; Intel Macs take the PyTorch path Linux and Windows use.

## Known limits

- Local AI features that use ONNX Runtime (voice, song analysis, index, columns) need `libonnxruntime.dylib` 1.27 or
  newer (ort asks for API 27). The newest Intel Mac build Microsoft published is 1.23.2, so these fail until one is
  built from source. Rendering video, pages, 2D and 3D does not need it.
- `raea motion` runs on PyTorch on the CPU (slow, ~18 GB memory for a new line).
- MetalFX upscaling and hardware ray tracing are Apple-Silicon features; expect them off.

## Rebuild after pulling upstream

```sh
git fetch upstream && git rebase upstream/main && git submodule update --init
cargo build --release -p raea-cli --no-default-features                  # ~16 min incremental, needs nasm (brew)
for d in sdk/raea sdk/react/js sdk/react/tw; do (cd $d && bun install); done
git describe --tags | sed 's/^v//' > VERSION
```

The build folder (`target/`) is about 13 GB.
