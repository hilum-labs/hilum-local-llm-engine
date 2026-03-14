#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "hilum_llm.h"
#include "testing.h"

namespace fs = std::filesystem;

namespace {

struct runtime_config {
    std::string model_path;
    std::string chat_model_path;
    std::string embed_model_path;
    std::string mmproj_path;
    std::string image_path;
};

runtime_config g_cfg;

const char * env_or_null(const char * name) {
    const char * value = std::getenv(name);
    return (value && value[0] != '\0') ? value : nullptr;
}

std::string compiled_chat_model_path() {
#ifdef HILUM_TEST_CHAT_MODEL_PATH
    return HILUM_TEST_CHAT_MODEL_PATH;
#else
    return "";
#endif
}

std::string resolve_chat_model_path() {
    if (const char * path = env_or_null("HILUM_TEST_CHAT_MODEL")) {
        return path;
    }

    const std::string compiled = compiled_chat_model_path();
    if (!compiled.empty() && fs::exists(compiled)) {
        return compiled;
    }

    return "";
}

std::string read_optional_path(const char * env_name) {
    if (const char * path = env_or_null(env_name)) {
        return path;
    }
    return "";
}

bool load_model(const std::string & path, hilum_model ** out_model) {
    hilum_model_params params = hilum_model_default_params();
    params.n_gpu_layers = 0;
    return hilum_model_load(path.c_str(), params, out_model) == HILUM_OK;
}

bool create_context(hilum_model * model, hilum_context ** out_ctx, uint32_t n_ctx = 256, uint32_t n_batch = 64) {
    hilum_context_params params = hilum_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_threads = 1;
    params.n_seq_max = n_batch > 1 ? n_batch : 1;
    params.flash_attn = 0;
    params.draft_model = nullptr;
    params.draft_n_max = 4;
    return hilum_context_create(model, params, out_ctx) == HILUM_OK;
}

std::vector<int32_t> tokenize_or_empty(hilum_model * model, const std::string & text, bool add_special = true) {
    int32_t needed = hilum_tokenize(model, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, false);
    if (needed >= 0) {
        return {};
    }

    std::vector<int32_t> tokens(static_cast<size_t>(-needed), 0);
    int32_t written = hilum_tokenize(
        model,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        false);
    if (written <= 0) {
        return {};
    }

    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

std::string detokenize_or_empty(hilum_model * model, const std::vector<int32_t> & tokens) {
    int32_t needed = hilum_detokenize(
        model,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        nullptr,
        0);
    if (needed >= 0) {
        return "";
    }

    std::vector<char> buf(static_cast<size_t>(-needed), '\0');
    int32_t written = hilum_detokenize(
        model,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        buf.data(),
        static_cast<int32_t>(buf.size()));
    if (written <= 0) {
        return "";
    }

    return std::string(buf.data(), static_cast<size_t>(written));
}

void test_model_load_and_free(testing & t) {
    hilum_model * model = nullptr;
    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("model handle is non-null", model != nullptr);
    t.assert_true("model size is non-zero", model != nullptr && hilum_model_size(model) > 0);
    hilum_model_free(model);
}

void test_context_create_and_free(testing & t) {
    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx));
    t.assert_true("context handle is non-null", ctx != nullptr);
    t.assert_true("context size is non-zero", ctx != nullptr && hilum_context_size(ctx) > 0);

    hilum_context_kv_clear(ctx, 0);
    hilum_context_free(ctx);
    hilum_model_free(model);
}

void test_tokenize_and_detokenize(testing & t) {
    hilum_model * model = nullptr;
    t.assert_true("model loads", load_model(g_cfg.model_path, &model));

    const std::string text = "Hilum runtime smoke test.";
    const std::vector<int32_t> tokens = tokenize_or_empty(model, text);
    t.assert_true("tokenize returns tokens", !tokens.empty());

    const std::string detok = detokenize_or_empty(model, tokens);
    t.assert_true("detokenize returns text", !detok.empty());
    t.assert_true("detokenized text preserves payload", detok.find("Hilum") != std::string::npos);

    hilum_model_free(model);
}

void test_chat_template(testing & t) {
    if (g_cfg.chat_model_path.empty()) {
        t.log("chat template smoke skipped: no chat model configured");
        return;
    }

    hilum_model * model = nullptr;
    t.assert_true("chat model loads", load_model(g_cfg.chat_model_path, &model));
    if (!model) {
        return;
    }

    constexpr const char * messages = R"([{"role":"user","content":"Say hello."}])";
    int32_t needed = hilum_chat_template(model, messages, true, nullptr, 0);
    t.assert_true("chat template reports required buffer size", needed < 0);
    if (needed >= 0) {
        hilum_model_free(model);
        return;
    }

    std::vector<char> buf(static_cast<size_t>(-needed + 1), '\0');
    int32_t written = hilum_chat_template(model, messages, true, buf.data(), static_cast<int32_t>(buf.size()));
    t.assert_true("chat template renders prompt", written > 0);
    if (written > 0) {
        const std::string rendered(buf.data(), static_cast<size_t>(written));
        t.assert_true("rendered prompt includes user message", rendered.find("Say hello.") != std::string::npos);
    }

    hilum_model_free(model);
}

void test_sync_generation(testing & t) {
    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx));
    if (!model || !ctx) {
        hilum_context_free(ctx);
        hilum_model_free(model);
        return;
    }

    hilum_gen_params params = hilum_gen_default_params();
    params.max_tokens = 12;
    params.temperature = 0.1f;
    params.top_k = 1;
    params.top_p = 0.95f;
    params.seed = 1234;

    std::vector<char> buf(512, '\0');
    int32_t generated = 0;
    hilum_error err = hilum_generate(
        model,
        ctx,
        "Continue this sentence in a few words: The sky is",
        params,
        buf.data(),
        static_cast<int32_t>(buf.size()),
        &generated);

    t.assert_equal("sync generation succeeds", HILUM_OK, err);
    t.assert_true("sync generation emits tokens", generated > 0);
    t.assert_true("sync generation returns text", std::strlen(buf.data()) > 0);

    const hilum_perf_data perf = hilum_get_perf(ctx);
    t.assert_true("perf tracks prompt eval or generation", perf.prompt_tokens >= 0 && perf.generated_tokens >= 0);

    hilum_context_free(ctx);
    hilum_model_free(model);
}

void test_extended_sampling_and_fit_helper(testing & t) {
    hilum_model_params fit_model = hilum_model_default_params();
    fit_model.n_gpu_layers = 0;
    hilum_context_params fit_ctx = hilum_context_default_params();
    fit_ctx.n_threads = 1;
    const hilum_fit_status fit_status = hilum_fit_params(g_cfg.model_path.c_str(), &fit_model, &fit_ctx, 64);
    t.assert_true("fit helper returns a non-error status", fit_status == HILUM_FIT_SUCCESS || fit_status == HILUM_FIT_FAILURE);

    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx));
    if (!model || !ctx) {
        hilum_context_free(ctx);
        hilum_model_free(model);
        return;
    }

    static const char * dry_breakers[] = { "\n", ":" };

    hilum_gen_params params = hilum_gen_default_params();
    params.max_tokens = 8;
    params.temperature = 0.8f;
    params.top_k = 20;
    params.top_p = 0.95f;
    params.min_p = 0.02f;
    params.typical_p = 0.98f;
    params.top_n_sigma = 2.0f;
    params.dry_multiplier = 0.5f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 32;
    params.dry_sequence_breakers = dry_breakers;
    params.n_dry_sequence_breakers = 2;
    params.dynatemp_range = 0.3f;
    params.dynatemp_exponent = 1.0f;
    params.grammar = "root ::= \"A\" | \"B\"";
    params.grammar_lazy = true;
    params.seed = 123;

    std::vector<char> buf(64, '\0');
    int32_t generated = 0;
    hilum_error err = hilum_generate(
        model,
        ctx,
        "Reply with A or B only.",
        params,
        buf.data(),
        static_cast<int32_t>(buf.size()),
        &generated);

    t.assert_equal("extended sampler generation succeeds", HILUM_OK, err);
    t.assert_true("extended sampler emits constrained text", std::strlen(buf.data()) > 0);

    hilum_context_kv_clear(ctx, 0);
    params.grammar = nullptr;
    params.grammar_lazy = false;
    params.mirostat = 2;
    params.mirostat_tau = 5.0f;
    params.mirostat_eta = 0.1f;
    params.adaptive_p_target = -1.0f;

    std::fill(buf.begin(), buf.end(), '\0');
    generated = 0;
    err = hilum_generate(
        model,
        ctx,
        "Continue with one short word:",
        params,
        buf.data(),
        static_cast<int32_t>(buf.size()),
        &generated);

    t.assert_equal("mirostat generation succeeds", HILUM_OK, err);
    t.assert_true("mirostat generation emits text", std::strlen(buf.data()) > 0);

    hilum_context_free(ctx);
    hilum_model_free(model);
}

struct stream_capture {
    int callbacks = 0;
    std::string text;
};

bool collect_first_token_then_cancel(const char * token, int32_t token_len, void * user_data) {
    auto * capture = static_cast<stream_capture *>(user_data);
    capture->callbacks++;
    if (token && token_len > 0) {
        capture->text.append(token, static_cast<size_t>(token_len));
    }
    return false;
}

void test_streaming_generation_and_cancellation(testing & t) {
    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx));
    if (!model || !ctx) {
        hilum_context_free(ctx);
        hilum_model_free(model);
        return;
    }

    hilum_gen_params params = hilum_gen_default_params();
    params.max_tokens = 16;
    params.temperature = 0.1f;
    params.top_k = 1;
    params.seed = 7;

    stream_capture capture;
    hilum_error err = hilum_generate_stream(
        model,
        ctx,
        "Write one short word:",
        params,
        collect_first_token_then_cancel,
        &capture);

    t.assert_equal("stream cancellation returns cancelled", HILUM_ERR_CANCELLED, err);
    t.assert_true("stream callback received at least one token", capture.callbacks >= 1);
    t.assert_true("stream callback accumulated text", !capture.text.empty());

    hilum_cancel(ctx);
    hilum_cancel_clear(ctx);

    hilum_context_free(ctx);
    hilum_model_free(model);
}

bool cancel_on_first_progress(int32_t tokens_processed, int32_t tokens_total, void * user_data) {
    auto * invoked = static_cast<int *>(user_data);
    (*invoked)++;
    return tokens_processed < tokens_total;
}

void test_prompt_progress_cancellation(testing & t) {
    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx, 128, 8));
    if (!model || !ctx) {
        hilum_context_free(ctx);
        hilum_model_free(model);
        return;
    }

    std::string prompt;
    for (int i = 0; i < 64; ++i) {
        prompt += "token ";
    }

    hilum_gen_params params = hilum_gen_default_params();
    params.max_tokens = 4;
    params.seed = 9;
    int progress_calls = 0;
    params.progress_callback = cancel_on_first_progress;
    params.progress_user_data = &progress_calls;

    std::vector<char> buf(64, '\0');
    int32_t generated = 0;
    hilum_error err = hilum_generate(
        model,
        ctx,
        prompt.c_str(),
        params,
        buf.data(),
        static_cast<int32_t>(buf.size()),
        &generated);

    t.assert_equal("progress callback cancellation returns cancelled", HILUM_ERR_CANCELLED, err);
    t.assert_true("progress callback was invoked", progress_calls >= 1);

    hilum_context_free(ctx);
    hilum_model_free(model);
}

void test_speculative_decoding(testing & t) {
    hilum_model * model = nullptr;
    hilum_model * draft = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("target model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("draft model loads", load_model(g_cfg.model_path, &draft));
    if (!model || !draft) {
        hilum_model_free(draft);
        hilum_model_free(model);
        return;
    }

    hilum_context_params params = hilum_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 64;
    params.n_threads = 1;
    params.draft_model = draft;
    params.draft_n_max = 4;

    t.assert_equal("speculative context creates", HILUM_OK, hilum_context_create(model, params, &ctx));
    if (!ctx) {
        hilum_model_free(draft);
        hilum_model_free(model);
        return;
    }

    hilum_gen_params gen = hilum_gen_default_params();
    gen.max_tokens = 8;
    gen.temperature = 0.1f;
    gen.top_k = 1;
    gen.seed = 11;

    std::vector<char> buf(256, '\0');
    int32_t generated = 0;
    hilum_error err = hilum_generate(
        model,
        ctx,
        "The quick brown fox",
        gen,
        buf.data(),
        static_cast<int32_t>(buf.size()),
        &generated);

    t.assert_equal("speculative generation succeeds", HILUM_OK, err);
    t.assert_true("speculative path emits tokens", generated > 0);

    hilum_context_free(ctx);
    hilum_model_free(draft);
    hilum_model_free(model);
}

void test_embeddings(testing & t) {
    t.assert_equal("null embedding dimension is zero", 0, hilum_emb_dimension(nullptr));

    hilum_emb_params invalid_params = {};
    invalid_params.pooling_type = -1;
    hilum_emb_ctx * invalid_ctx = nullptr;
    t.assert_equal("embedding context validates null model", HILUM_ERR_INVALID_HANDLE,
        hilum_emb_context_create(nullptr, invalid_params, &invalid_ctx));

    if (g_cfg.embed_model_path.empty()) {
        t.log("embedding smoke skipped: no embedding-capable GGUF configured via HILUM_TEST_EMBED_MODEL");
        return;
    }

    hilum_model * model = nullptr;
    hilum_emb_ctx * emb_ctx = nullptr;

    t.assert_true("embedding model loads", load_model(g_cfg.embed_model_path, &model));
    if (!model) {
        return;
    }

    const int32_t dim = hilum_emb_dimension(model);
    t.assert_true("embedding dimension is positive", dim > 0);

    hilum_emb_params params = {};
    params.n_ctx = 256;
    params.n_batch = 64;
    params.n_threads = 1;
    params.pooling_type = 1;
    t.assert_equal("embedding context creates", HILUM_OK, hilum_emb_context_create(model, params, &emb_ctx));
    if (!emb_ctx) {
        hilum_model_free(model);
        return;
    }

    const std::vector<int32_t> tokens = tokenize_or_empty(model, "Embedding smoke test.");
    t.assert_true("embedding input tokenizes", !tokens.empty());
    if (tokens.empty()) {
        hilum_emb_context_free(emb_ctx);
        hilum_model_free(model);
        return;
    }

    std::vector<float> embedding(static_cast<size_t>(dim), 0.0f);
    t.assert_equal("single embedding succeeds", HILUM_OK,
        hilum_embed(emb_ctx, model, tokens.data(), static_cast<int32_t>(tokens.size()), embedding.data(), dim));
    t.assert_true("single embedding has non-zero values",
        std::any_of(embedding.begin(), embedding.end(), [](float value) { return value != 0.0f; }));

    const std::vector<int32_t> tokens_b = tokenize_or_empty(model, "Second embedding input.");
    t.assert_true("second embedding input tokenizes", !tokens_b.empty());
    if (tokens_b.empty()) {
        hilum_emb_context_free(emb_ctx);
        hilum_model_free(model);
        return;
    }

    std::vector<float> embedding_b(static_cast<size_t>(dim), 0.0f);
    const int32_t * token_arrays[2] = { tokens.data(), tokens_b.data() };
    const int32_t token_counts[2] = {
        static_cast<int32_t>(tokens.size()),
        static_cast<int32_t>(tokens_b.size()),
    };
    float * outputs[2] = { embedding.data(), embedding_b.data() };

    t.assert_equal("batch embedding succeeds", HILUM_OK,
        hilum_embed_batch(emb_ctx, model, token_arrays, token_counts, 2, outputs, dim));
    t.assert_true("batch embedding fills second output",
        std::any_of(embedding_b.begin(), embedding_b.end(), [](float value) { return value != 0.0f; }));

    hilum_emb_context_free(emb_ctx);
    hilum_model_free(model);
}

void test_mtmd_contract(testing & t) {
    t.assert_true("null mtmd reports no vision support", !hilum_mtmd_supports_vision(nullptr));

    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;
    hilum_mtmd * mtmd = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx));
    if (!model || !ctx) {
        hilum_context_free(ctx);
        hilum_model_free(model);
        return;
    }

    hilum_mtmd_params params = {};
    params.use_gpu = false;
    params.n_threads = 1;

    t.assert_equal("invalid projector fails cleanly", HILUM_ERR_VISION_FAILED,
        hilum_mtmd_load(model, "/definitely/missing/projector.gguf", params, &mtmd));
    t.assert_true("failed mtmd load leaves handle null", mtmd == nullptr);

    const hilum_image image = { nullptr, 0 };
    hilum_gen_params gen = hilum_gen_default_params();
    gen.max_tokens = 4;

    t.assert_equal("vision generate rejects null mtmd", HILUM_ERR_INVALID_HANDLE,
        hilum_generate_vision(model, ctx, nullptr, "describe", &image, 1, gen, nullptr, 0, nullptr));
    t.assert_equal("vision stream rejects null mtmd", HILUM_ERR_INVALID_HANDLE,
        hilum_generate_vision_stream(model, ctx, nullptr, "describe", &image, 1, gen, collect_first_token_then_cancel, nullptr));

    if (!g_cfg.mmproj_path.empty() && !g_cfg.image_path.empty() &&
        fs::exists(g_cfg.mmproj_path) && fs::exists(g_cfg.image_path)) {
        std::vector<uint8_t> image_buf;
        {
            std::ifstream image_file(g_cfg.image_path, std::ios::binary);
            image_buf.assign(
                std::istreambuf_iterator<char>(image_file),
                std::istreambuf_iterator<char>());
        }

        hilum_error load_err = hilum_mtmd_load(model, g_cfg.mmproj_path.c_str(), params, &mtmd);
        t.assert_equal("configured mtmd loads", HILUM_OK, load_err);
        if (load_err == HILUM_OK && mtmd != nullptr && !image_buf.empty()) {
            t.assert_true("configured mtmd reports vision support", hilum_mtmd_supports_vision(mtmd));
            const hilum_image real_image = { image_buf.data(), image_buf.size() };
            std::vector<char> buf(256, '\0');
            int32_t generated = 0;
            hilum_error vision_err = hilum_generate_vision(
                model, ctx, mtmd,
                "<__media__>\nDescribe this image briefly.",
                &real_image, 1, gen,
                buf.data(), static_cast<int32_t>(buf.size()), &generated);
            t.assert_equal("configured vision generate succeeds", HILUM_OK, vision_err);
            t.assert_true("configured vision generate emits text", generated >= 0);
            hilum_mtmd_free(mtmd);
        }
    } else {
        t.log("vision smoke skipped: HILUM_TEST_MMPROJ and HILUM_TEST_IMAGE not set");
    }

    hilum_context_free(ctx);
    hilum_model_free(model);
}

void test_quantize_smoke(testing & t) {
    hilum_quantize_params params = hilum_quantize_default_params();
    params.allow_requantize = true;
    params.nthread = 1;

    const fs::path out_path = fs::temp_directory_path() / "hilum-phase0-quantized.gguf";
    std::error_code ec;
    fs::remove(out_path, ec);

    const hilum_error err = hilum_quantize(
        g_cfg.model_path.c_str(),
        out_path.string().c_str(),
        params);
    t.assert_equal("quantize smoke succeeds", HILUM_OK, err);
    t.assert_true("quantized file exists", fs::exists(out_path));
    t.assert_true("quantized file is non-empty", fs::exists(out_path) && fs::file_size(out_path) > 0);

    fs::remove(out_path, ec);
}

void test_warmup_and_benchmark(testing & t) {
    hilum_model * model = nullptr;
    hilum_context * ctx = nullptr;

    t.assert_true("model loads", load_model(g_cfg.model_path, &model));
    t.assert_true("context creates", model != nullptr && create_context(model, &ctx));
    if (!model || !ctx) {
        hilum_context_free(ctx);
        hilum_model_free(model);
        return;
    }

    t.assert_equal("warmup succeeds", HILUM_OK, hilum_warmup(model, ctx));

    hilum_benchmark_result result = {};
    hilum_benchmark_params params = hilum_benchmark_default_params();
    params.prompt_tokens = 8;
    params.generate_tokens = 4;
    params.iterations = 1;

    t.assert_equal("benchmark succeeds", HILUM_OK, hilum_benchmark(model, ctx, params, &result));
    t.assert_equal("benchmark completes one iteration", 1, result.iterations);
    t.assert_true("benchmark total time is non-negative", result.total_ms >= 0.0);

    hilum_context_free(ctx);
    hilum_model_free(model);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: test-hilum-runtime <model-path> [filter]\n";
        return 2;
    }

    g_cfg.model_path = argv[1];
    g_cfg.chat_model_path = resolve_chat_model_path();
    g_cfg.embed_model_path = read_optional_path("HILUM_TEST_EMBED_MODEL");
    g_cfg.mmproj_path = read_optional_path("HILUM_TEST_MMPROJ");
    g_cfg.image_path = read_optional_path("HILUM_TEST_IMAGE");

    testing t(std::cout);
    if (argc >= 3) {
        t.set_filter(argv[2]);
    }

    t.test("model load and free", test_model_load_and_free);
    t.test("context create and free", test_context_create_and_free);
    t.test("tokenize and detokenize", test_tokenize_and_detokenize);
    t.test("chat template", test_chat_template);
    t.test("sync generation", test_sync_generation);
    t.test("extended sampling and fit helper", test_extended_sampling_and_fit_helper);
    t.test("streaming generation and cancellation", test_streaming_generation_and_cancellation);
    t.test("prompt progress cancellation", test_prompt_progress_cancellation);
    t.test("speculative decoding", test_speculative_decoding);
    t.test("embeddings", test_embeddings);
    t.test("mtmd contract", test_mtmd_contract);
    t.test("quantize smoke", test_quantize_smoke);
    t.test("warmup and benchmark", test_warmup_and_benchmark);

    return t.summary();
}
