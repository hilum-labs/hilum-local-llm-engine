#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "hilum_llm.h"
#include "testing.h"

static void test_api_version(testing & t) {
    const uint32_t expected =
        (static_cast<uint32_t>(HILUM_API_VERSION_MAJOR) << 16) |
        (static_cast<uint32_t>(HILUM_API_VERSION_MINOR) << 8) |
        static_cast<uint32_t>(HILUM_API_VERSION_PATCH);

    t.assert_equal("api version matches header macros", expected, hilum_api_version());
}

static void test_metadata(testing & t) {
    const char * backend = hilum_backend_info();
    const char * version = hilum_backend_version();

    t.assert_true("backend info is non-null", backend != nullptr);
    t.assert_true("backend info is non-empty", backend != nullptr && std::strlen(backend) > 0);
    t.assert_true("backend version is non-null", version != nullptr);
    t.assert_true("backend version contains hilum engine marker",
        version != nullptr && std::string(version).find("hilum-local-llm-engine") != std::string::npos);
}

static void test_error_strings(testing & t) {
    for (int err = HILUM_OK; err <= HILUM_ERR_VISION_FAILED; ++err) {
        const char * msg = hilum_error_str(static_cast<hilum_error>(err));
        t.assert_true("known error strings are non-empty", msg != nullptr && std::strlen(msg) > 0);
    }

    t.assert_equal("unknown errors fall back to stable string",
        std::string("unknown error"),
        std::string(hilum_error_str(static_cast<hilum_error>(999))));
}

static void test_default_params(testing & t) {
    const hilum_model_params model = hilum_model_default_params();
    t.assert_equal("default n_gpu_layers uses auto", -1, model.n_gpu_layers);
    t.assert_true("default model mmap is enabled", model.use_mmap);

    const hilum_context_params ctx = hilum_context_default_params();
    t.assert_equal("default n_ctx is model default", 0u, ctx.n_ctx);
    t.assert_equal("default n_batch is model default", 0u, ctx.n_batch);
    t.assert_equal("default n_threads is auto", 0u, ctx.n_threads);
    t.assert_equal("flash attention defaults off", 0, ctx.flash_attn);
    t.assert_equal("default K cache type tracks llama defaults", -1, ctx.type_k);
    t.assert_equal("default V cache type tracks llama defaults", -1, ctx.type_v);
    t.assert_equal("default parallel sequence count uses engine default", 0u, ctx.n_seq_max);
    t.assert_true("speculative decoding is disabled by default", ctx.draft_model == nullptr);
    t.assert_equal("default speculative draft width is stable", 16, ctx.draft_n_max);

    const hilum_gen_params gen = hilum_gen_default_params();
    t.assert_equal("default generation max tokens", 512, gen.max_tokens);
    t.assert_equal("default generation top-k", 40, gen.top_k);
    t.assert_true("default temperature is 0.7f", gen.temperature == 0.7f);
    t.assert_true("default top-p is 0.9f", gen.top_p == 0.9f);
    t.assert_true("default repeat penalty is 1.1f", gen.repeat_penalty == 1.1f);
    t.assert_true("default grammar is null", gen.grammar == nullptr);
    t.assert_true("default grammar root is null", gen.grammar_root == nullptr);
    t.assert_true("default stop sequences are null", gen.stop_sequences == nullptr);
    t.assert_equal("default stop sequence count", 0, gen.n_stop_sequences);
    t.assert_equal("default min_keep is disabled", 0, gen.min_keep);
    t.assert_true("default min_p is disabled", gen.min_p == 0.0f);
    t.assert_true("default typical_p is disabled", gen.typical_p == 1.0f);
    t.assert_true("default top_n_sigma is disabled", gen.top_n_sigma == -1.0f);
    t.assert_true("default xtc probability is disabled", gen.xtc_probability == 0.0f);
    t.assert_true("default dry multiplier is disabled", gen.dry_multiplier == 0.0f);
    t.assert_true("default adaptive-p target is disabled", gen.adaptive_p_target == -1.0f);
    t.assert_equal("default mirostat is disabled", 0, gen.mirostat);
    t.assert_true("default grammar lazy is disabled", !gen.grammar_lazy);

    const hilum_quantize_params quant = hilum_quantize_default_params();
    t.assert_equal("default quantization type is Q4_K_M", 15, quant.ftype);
    t.assert_equal("default quantization threads use auto", 0, quant.nthread);
    t.assert_true("requantization is disabled by default", !quant.allow_requantize);
    t.assert_true("output tensor quantization remains enabled", quant.quantize_output_tensor);
    t.assert_true("pure quantization remains disabled", !quant.pure);

    const hilum_benchmark_params bench = hilum_benchmark_default_params();
    t.assert_equal("default benchmark prompt tokens", 128, bench.prompt_tokens);
    t.assert_equal("default benchmark generation tokens", 64, bench.generate_tokens);
    t.assert_equal("default benchmark iterations", 3, bench.iterations);
}

static void test_utility_apis(testing & t) {
    t.assert_true("optimal thread count is always positive", hilum_optimal_thread_count() >= 1);

    hilum_cancel(nullptr);
    hilum_cancel_clear(nullptr);
    hilum_model_free(nullptr);
    hilum_context_free(nullptr);
    hilum_mtmd_free(nullptr);
    hilum_emb_context_free(nullptr);
    hilum_log_set_level(HILUM_LOG_WARN);
    hilum_log_set(nullptr, nullptr);

    const hilum_perf_data perf = hilum_get_perf(nullptr);
    t.assert_true("null perf query returns zeroed prompt eval",
        perf.prompt_eval_ms == 0.0 && perf.generation_ms == 0.0);

    hilum_benchmark_result out = {};
    t.assert_equal("benchmark validates null handles", HILUM_ERR_INVALID_HANDLE,
        hilum_benchmark(nullptr, nullptr, hilum_benchmark_default_params(), &out));
    hilum_model_params model_params = hilum_model_default_params();
    hilum_context_params ctx_params = hilum_context_default_params();
    t.assert_equal("fit helper validates null inputs", HILUM_FIT_ERROR,
        hilum_fit_params(nullptr, &model_params, &ctx_params, 0));
}

static void test_json_schema_bridge(testing & t) {
    constexpr const char * schema = R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})";

    const int32_t needed = hilum_json_schema_to_grammar(schema, nullptr, 0);
    t.assert_true("schema conversion reports required buffer size", needed < 0);

    std::vector<char> buf(static_cast<size_t>(-needed), '\0');
    const int32_t written = hilum_json_schema_to_grammar(schema, buf.data(), static_cast<int32_t>(buf.size()));
    t.assert_true("schema conversion succeeds with enough space", written > 0);
    t.assert_equal("grammar length excludes terminator", written, static_cast<int32_t>(std::strlen(buf.data())));

    const std::string grammar(buf.data(), static_cast<size_t>(written));
    t.assert_true("grammar contains root rule", grammar.find("root") != std::string::npos);
    t.assert_equal("invalid json schema returns 0", 0, hilum_json_schema_to_grammar("{", nullptr, 0));
    t.assert_equal("null json schema returns 0", 0, hilum_json_schema_to_grammar(nullptr, nullptr, 0));
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("api version", test_api_version);
    t.test("metadata", test_metadata);
    t.test("error strings", test_error_strings);
    t.test("default params", test_default_params);
    t.test("utility apis", test_utility_apis);
    t.test("json schema bridge", test_json_schema_bridge);

    return t.summary();
}
