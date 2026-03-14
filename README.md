# Hilum LLM Engine

On-Device LLM inference engine, built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform: iOS | Android | macOS | Linux](https://img.shields.io/badge/platform-iOS%20%7C%20Android%20%7C%20macOS%20%7C%20Linux-lightgrey)
[![CI](https://img.shields.io/github/actions/workflow/status/hilum-labs/hilum-local-llm-engine/build.yml?branch=main&label=CI)](https://github.com/hilum-labs/hilum-local-llm-engine/actions)

## What is this

A maintained fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for **on-device inference**. This is the core C++ engine that powers the Hilum ecosystem. You typically consume it through one of these packages:

| Package | Runtime | Platform | Install |
| --- | --- | --- | --- |
| [`local-llm`](https://www.npmjs.com/package/local-llm) | Node.js | macOS, Linux | `npm install local-llm` |
| [`local-llm-rn`](https://www.npmjs.com/package/local-llm-rn) | React Native | iOS (Metal), Android (Vulkan) | `npm install local-llm-rn` |

> If you're looking to **use** the engine for inference, head to the package READMEs above. This README covers the engine internals, build process, and our changes from upstream.

We manually forward-port upstream llama.cpp changes and add mobile-specific optimizations on top. The goal is broad GGUF compatibility, but parity is maintained through targeted subsystem ports and validation rather than by fast-forward merging upstream history.

## Building

```bash
cmake -B build \
  -DGGML_VULKAN=ON \
  -DGGML_VULKAN_VMA=ON \
  -DGGML_VULKAN_BUILD_ADRENO_SHADERS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

See [docs/build.md](docs/build.md) for full build options and [docs/android.md](docs/android.md) for Android cross-compilation.

To build the Hilum C API and its engine-level smoke tests:

```bash
cmake -B build \
  -DHILUM_BUILD_LIB=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_METAL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --target test-hilum-api test-hilum-runtime
ctest --test-dir build -R 'test-hilum-(api|runtime)' --output-on-failure
```

`test-hilum-runtime` uses the tiny GGUF fixture already wired into upstream tests. Positive multimodal and embedding smoke paths can be enabled by setting `HILUM_TEST_MMPROJ`, `HILUM_TEST_IMAGE`, and `HILUM_TEST_EMBED_MODEL`.

## What we changed

All upstream llama.cpp functionality is preserved. Our additions:

### Adreno shader variants

Qualcomm Adreno GPUs (found in most Android phones) get optimized SPIR-V compute shaders that better match their hardware architecture.

```cmake
-DGGML_VULKAN_BUILD_ADRENO_SHADERS=ON
```

At runtime, the engine detects Qualcomm vendor ID (`VK_VENDOR_ID_QUALCOMM`) and swaps in the Adreno-optimized shader variant automatically. No user configuration needed.

**Files**: `ggml/src/ggml-vulkan/CMakeLists.txt`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-vulkan/vulkan-shaders/`

### Vulkan Memory Allocator (VMA)

Efficient GPU memory management for Android devices using the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) library. Reduces memory fragmentation and allocation overhead during inference.

```cmake
-DGGML_VULKAN=ON
-DGGML_VULKAN_VMA=ON
```

### Bayesian device tuning

A per-device parameter tuning script that uses Bayesian optimization (Gaussian Process) to find optimal inference settings for a specific device.

```bash
python scripts/tune_device.py \
  --model model.gguf \
  --n-iter 30 \
  --output device-config.json
```

Tunes `n_threads`, `n_batch`, `n_gpu_layers`, and `flash_attn` to maximize tokens/second on the target hardware. Outputs a JSON config that can be fed back into the engine.

### Mobile build configurations

Pre-configured CMake, Gradle, and CocoaPods build settings tuned for:

- **iOS**: Metal + Accelerate + ARM NEON, BF16 support, embedded Metal library
- **Android**: Vulkan + Adreno shaders + CPU variant dispatch (7 ARM variants from armv8.0 to armv9.2), KleidiAI, OpenMP

## Benchmarks

Measured with `llama-bench`, default sampling, GPU offload where available. Numbers are tokens/second.

| Model | Quant | Device | Prompt (tok/s) | Generation (tok/s) |
| --- | --- | --- | ---: | ---: |
| Llama 3.2 3B | Q4_K_M | iPhone 15 Pro (Metal) | ~320 | ~42 |
| Llama 3.2 3B | Q4_K_M | Pixel 8 (Vulkan) | ~190 | ~28 |
| Llama 3.2 3B | Q4_K_M | MacBook Pro M2 | ~860 | ~80 |
| Phi-3 Mini 3.8B | Q4_K_M | iPhone 15 Pro (Metal) | ~280 | ~36 |
| SmolLM2 1.7B | Q4_K_M | Pixel 7a (Vulkan) | ~240 | ~35 |

> **Note**: These are approximate numbers from internal testing. Results vary with OS version, thermal state, and background load. Run `scripts/tune_device.py` on your target hardware for precise numbers.

## Recommended models for mobile

| Use case | Model | Quant | Size | Notes |
| --- | --- | --- | ---: | --- |
| Fast assistant | SmolLM2 1.7B | Q4_K_M | ~1.0 GB | Best speed/quality for constrained devices |
| General chat | Llama 3.2 3B | Q4_K_M | ~1.8 GB | Strong quality, runs well on 2023+ flagships |
| General chat (higher quality) | Llama 3.2 3B | Q8_0 | ~3.2 GB | Noticeably better output, needs 6+ GB RAM |
| Coding | Qwen 2.5 Coder 3B | Q4_K_M | ~1.9 GB | Competitive code generation at small size |
| Multimodal | MiniCPM-V 2.6 | Q4_K_M | ~4.5 GB | Vision + language, needs 8+ GB RAM |

All models must be in [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) format. The engine supports 100+ architectures including LLaMA, Mistral, Mixtral, Phi, Qwen, Gemma, DeepSeek, Yi, and many more. Full list: [llama.cpp supported models](https://github.com/ggml-org/llama.cpp#description).

Quantization levels from 1.5-bit to 8-bit are supported. For mobile we recommend **Q4_K_M** as the default (best balance of quality, speed, and size).

## Architecture

```
Node.js:   C++ engine <- N-API addon <- TypeScript (local-llm)
iOS:       C++ engine <- Obj-C++ Turbo Module <- TypeScript (local-llm-rn)
Android:   C++ engine <- JNI C++ / Kotlin Turbo Module <- TypeScript (local-llm-rn)
```

The C++ engine is compiled from source on each platform:
- **iOS**: Via CocoaPods, using the `local-llm-rn.podspec`
- **Android**: Via Gradle + CMake, with CPU variant `.so` files loaded at runtime
- **Node.js**: Via node-gyp / cmake-js

## Mobile GPU backends

| Platform | GPU Backend | Key Features |
| --- | --- | --- |
| iOS | Metal | Embedded shaders, BF16, Accelerate BLAS |
| Android | Vulkan | Adreno shader variants, VMA, dynamic backend loading |

On Android, CPU inference uses variant dispatch (`GGML_CPU_ALL_VARIANTS`) to automatically select the best instruction set for the device (NEON, dotprod, i8mm, SVE, etc.).

## Device tuning (optional)

```bash
pip install scikit-optimize numpy
python scripts/tune_device.py \
  --model model.gguf \
  --n-iter 30 \
  --output config.json
```

## Upstream compatibility

This fork follows [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) through manual forward-porting. Our Hilum-specific changes are intentionally kept concentrated in the wrapper and mobile backend layers, but upstream reconciliation is a subsystem-by-subsystem engineering task, not a trivial merge.

- Upstream API changelogs: [libllama](https://github.com/ggml-org/llama.cpp/issues/9289) | [server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on the PR process, coding standards, and how to run benchmarks with `llama-bench` before submitting.

## License

MIT -- same as upstream llama.cpp. See [LICENSE](LICENSE) for details.

## Contact

Questions, feedback, or partnership inquiries: [info@hilumlabs.com](mailto:info@hilumlabs.com)

## Acknowledgments

Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and the ggml community. We are grateful for their work making high-performance LLM inference accessible.
