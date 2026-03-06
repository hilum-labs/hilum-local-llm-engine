# Contributing

Thanks for your interest in contributing to Hilum LLM Engine. We're happy to have you here.

Please take a moment to review this document before submitting your first pull request. We also strongly recommend checking for [open issues](https://github.com/hilum-labs/hilum-local-llm-engine/issues) and [pull requests](https://github.com/hilum-labs/hilum-local-llm-engine/pulls) to see if someone else is working on something similar.

## About this repository

This is a maintained fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for on-device inference. It powers the [local-llm](https://www.npmjs.com/package/local-llm) and [local-llm-rn](https://www.npmjs.com/package/local-llm-rn) packages.

Our changes from upstream are additive and isolated behind build flags:

- Adreno shader variants for Qualcomm GPUs
- Vulkan Memory Allocator (VMA) integration
- Bayesian device tuning script
- Mobile build configurations (iOS Metal, Android Vulkan)

## Structure

```
ggml/                  # Core tensor library
├── src/ggml-vulkan/   # Vulkan backend (Adreno shaders live here)
├── src/ggml-metal/    # Metal backend
src/                   # llama.cpp model loading and inference
include/               # Public C API headers
scripts/               # Utility scripts (tune_device.py, etc.)
docs/                  # Build and backend documentation
```

## Development

### Fork this repo

You can fork this repo by clicking the fork button in the top right corner of the [repository page](https://github.com/hilum-labs/hilum-local-llm-engine).

### Clone on your local machine

```bash
git clone https://github.com/your-username/hilum-local-llm-engine.git
```

### Navigate to project directory

```bash
cd hilum-local-llm-engine
```

### Create a new branch

```bash
git checkout -b my-new-branch
```

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)
```

For GPU-accelerated builds, see [docs/build.md](docs/build.md).

## Testing

Before submitting your PR, verify that your changes don't break anything:

### Build test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)
```

### Performance check

Run benchmarks to ensure your changes don't regress performance:

```bash
./build/bin/llama-bench -m model.gguf
```

### Perplexity check

If you've modified model evaluation code, verify output quality:

```bash
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw
```

### Backend ops test

If you've modified a `ggml` operator, run the backend consistency tests:

```bash
./build/bin/test-backend-ops
```

## Pull requests

When you're ready to submit:

1. **Search first.** Check for existing PRs to avoid duplicate work.
2. **One thing per PR.** Keep changes focused. Don't combine unrelated fixes.
3. **Test your changes.** See the [Testing](#testing) section above.
4. **Describe what you did.** Explain the "why" in your PR description, not just the "what."

### After submitting

- Expect review feedback. We may request changes to match the project's coding standards.
- If your PR goes stale, rebase on `main` to bring it back to attention.
- Consider enabling write access on your branch so reviewers can push fixes directly.

## Commit convention

We follow a simple commit convention:

```
<module> : <description>
```

Examples:

```
vulkan : add Adreno 750 shader variant
metal : fix BF16 matmul for A17 Pro
scripts : improve tune_device.py convergence
docs : update Android build instructions
```

Pick a module that matches the area you changed (e.g. `vulkan`, `metal`, `ggml`, `scripts`, `docs`, `ci`, `build`).

## Coding guidelines

- **Keep it simple.** Avoid fancy STL constructs, heavy templates, or unnecessary abstractions.
- **No new dependencies** unless absolutely necessary.
- **Cross-platform.** Always consider macOS, Linux, iOS, and Android.
- **Formatting:** 4 spaces, no trailing whitespace, brackets on the same line. Use `clang-format` (v15+) when in doubt.
- **Types:** Use sized integers (`int32_t`) in public APIs.
- **Naming:** `snake_case` everywhere. Optimize for longest common prefix.

```cpp
// Good
int number_small;
int number_big;

// Bad
int small_number;
int big_number;
```

- **Enums:** uppercase values, prefixed with the enum name.

```cpp
enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,
    LLAMA_VOCAB_TYPE_SPM  = 1,
    LLAMA_VOCAB_TYPE_BPE  = 2,
};
```

- **Functions:** follow the `<class>_<action>_<noun>` pattern.

```cpp
llama_model_init();
llama_sampler_chain_remove();
llama_sampler_get_seed();
```

For anything not covered here, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).

## Documentation

- When you figure something out by reading source code, consider adding a comment or doc update so the next person doesn't have to.
- If you notice incorrect or outdated documentation, please update it.

## Questions?

Open an [issue](https://github.com/hilum-labs/hilum-local-llm-engine/issues), start a [discussion](https://github.com/hilum-labs/hilum-local-llm-engine/discussions), or reach out at [info@hilumlabs.com](mailto:info@hilumlabs.com). We're happy to help.
