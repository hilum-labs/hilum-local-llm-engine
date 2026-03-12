/*
 * hilum_llm.cpp — Implementation of the Hilum shared C API.
 *
 * This file consolidates the inference logic previously duplicated across
 * addon.cpp (Node.js N-API) and LocalLLM.mm (React Native Turbo Module)
 * into a single, platform-neutral implementation.
 */

#ifndef HILUM_BUILD
#define HILUM_BUILD
#endif

#include "hilum_llm.h"

#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <chrono>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <fstream>
#endif

#include "llama.h"
#include "../mtmd/mtmd.h"
#include "../mtmd/mtmd-helper.h"
#include "../common/json-schema-to-grammar.h"
#include <nlohmann/json.hpp>

/* ── Backend initialization (once) ─────────────────────────────────────────── */

static std::once_flag g_backend_flag;

static void ensure_backend() {
    std::call_once(g_backend_flag, []() { llama_backend_init(); });
}

/* ── Live handle registry (double-free protection) ─────────────────────────── */

static std::mutex g_handle_mutex;
static std::unordered_set<void *> g_live_handles;

static void register_handle(void * ptr) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    g_live_handles.insert(ptr);
}

static bool unregister_handle(void * ptr) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    return g_live_handles.erase(ptr) > 0;
}

/* ── Thread-local token buffer ─────────────────────────────────────────────── */

struct TokenBuf {
    char small[256];
    std::vector<char> big;
    const char * ptr;
    int32_t len;

    void decode(const llama_vocab * vocab, llama_token token) {
        int32_t n = llama_token_to_piece(vocab, token, small, sizeof(small), 0, false);
        if (n >= 0) {
            ptr = small;
            len = n;
        } else {
            big.resize(-n);
            llama_token_to_piece(vocab, token, big.data(), big.size(), 0, false);
            ptr = big.data();
            len = -n;
        }
    }
};

static thread_local TokenBuf t_token_buf;

/* ── Thread-local error message ────────────────────────────────────────────── */

static thread_local std::string g_last_error;

static void set_error(const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_last_error = buf;
}

HILUM_API const char * hilum_last_error(void) {
    return g_last_error.c_str();
}

static std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    t_token_buf.decode(vocab, token);
    return std::string(t_token_buf.ptr, t_token_buf.len);
}

static void token_append(const llama_vocab * vocab, llama_token token, std::string & out) {
    t_token_buf.decode(vocab, token);
    out.append(t_token_buf.ptr, t_token_buf.len);
}

/* ── Opaque handle structs ─────────────────────────────────────────────────── */

struct hilum_model {
    llama_model * llm;
};

struct hilum_context {
    llama_context * llm;
    llama_model   * model_llm;       // back-ref (non-owning)

    llama_context * draft_llm;       // NULL if no speculative decoding
    llama_model   * draft_model_llm; // non-owning ref to draft model
    int32_t         draft_n_max;     // max draft tokens per step

    std::atomic<bool> cancel_flag{false};  // cross-thread cancellation
};

struct hilum_mtmd {
    mtmd_context * ctx;
};

struct hilum_emb_ctx {
    llama_context * llm;
    llama_model   * model_llm; // back-ref (non-owning)
};

/* ── Sampler cache ─────────────────────────────────────────────────────────── */

struct SamplerParams {
    int32_t  top_k             = 40;
    float    top_p             = 0.9f;
    float    temperature       = 0.7f;
    float    repeat_penalty    = 1.1f;
    float    frequency_penalty = 0.0f;
    float    presence_penalty  = 0.0f;
    uint32_t seed              = LLAMA_DEFAULT_SEED;

    bool operator==(const SamplerParams & o) const {
        return top_k == o.top_k && top_p == o.top_p &&
               temperature == o.temperature &&
               repeat_penalty == o.repeat_penalty &&
               frequency_penalty == o.frequency_penalty &&
               presence_penalty == o.presence_penalty &&
               seed == o.seed;
    }
};

struct SamplerCache {
    llama_sampler * sampler = nullptr;
    SamplerParams   params;

    void clear() {
        if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
    }
};

static std::mutex g_sampler_cache_mutex;
static std::unordered_map<llama_context *, SamplerCache> g_sampler_cache;

static llama_sampler * create_sampler(
    const llama_vocab * vocab,
    int32_t top_k, float top_p, float temperature,
    float repeat_penalty, float frequency_penalty, float presence_penalty,
    uint32_t seed,
    const std::string & grammar_str = "",
    const std::string & grammar_root = "root",
    llama_context * cache_ctx = nullptr)
{
    if (cache_ctx && grammar_str.empty()) {
        SamplerParams cur{top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed};
        std::lock_guard<std::mutex> lock(g_sampler_cache_mutex);
        auto it = g_sampler_cache.find(cache_ctx);
        if (it != g_sampler_cache.end() && it->second.sampler && it->second.params == cur) {
            llama_sampler * cached = it->second.sampler;
            it->second.sampler = nullptr;
            return cached;
        }
    }

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(chain_params);

    if (!grammar_str.empty()) {
        llama_sampler * g = llama_sampler_init_grammar(vocab, grammar_str.c_str(), grammar_root.c_str());
        if (g) llama_sampler_chain_add(smpl, g);
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, repeat_penalty, frequency_penalty, presence_penalty));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

static void return_sampler(
    llama_context * cache_ctx, llama_sampler * smpl,
    const SamplerParams & params, bool has_grammar)
{
    if (!cache_ctx || has_grammar) {
        llama_sampler_free(smpl);
        return;
    }
    std::lock_guard<std::mutex> lock(g_sampler_cache_mutex);
    auto & entry = g_sampler_cache[cache_ctx];
    entry.clear();
    entry.sampler = smpl;
    entry.params  = params;
}

static void evict_sampler_cache(llama_context * ctx) {
    std::lock_guard<std::mutex> lock(g_sampler_cache_mutex);
    auto it = g_sampler_cache.find(ctx);
    if (it != g_sampler_cache.end()) {
        it->second.clear();
        g_sampler_cache.erase(it);
    }
}

/* ── Logging ───────────────────────────────────────────────────────────────── */

static hilum_log_callback g_log_cb = nullptr;
static void *             g_log_ud = nullptr;
static std::atomic<int>   g_log_min_level{0};

static void native_log_callback(enum ggml_log_level level, const char * text, void * /*ud*/) {
    if (!g_log_cb) return;
    if (static_cast<int>(level) < g_log_min_level.load(std::memory_order_relaxed)) return;
    g_log_cb(static_cast<hilum_log_level>(level), text, g_log_ud);
}

/* ── Helpers ───────────────────────────────────────────────────────────────── */

static SamplerParams to_sampler_params(const hilum_gen_params & p) {
    return SamplerParams{
        p.top_k, p.top_p, p.temperature,
        p.repeat_penalty, p.frequency_penalty, p.presence_penalty,
        p.seed
    };
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PUBLIC API                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ── Error string ──────────────────────────────────────────────────────────── */

HILUM_API const char * hilum_error_str(hilum_error err) {
    switch (err) {
        case HILUM_OK:               return "ok";
        case HILUM_ERR_LOAD_FAILED:  return "model load failed";
        case HILUM_ERR_CONTEXT_FAILED: return "context creation failed";
        case HILUM_ERR_TOKENIZE_FAILED: return "tokenization failed";
        case HILUM_ERR_DECODE_FAILED:  return "decode failed";
        case HILUM_ERR_ENCODE_FAILED:  return "encode failed";
        case HILUM_ERR_INVALID_HANDLE: return "invalid handle";
        case HILUM_ERR_OOM:            return "out of memory";
        case HILUM_ERR_GRAMMAR_FAILED: return "grammar parsing failed";
        case HILUM_ERR_QUANTIZE_FAILED: return "quantization failed";
        case HILUM_ERR_CANCELLED:      return "cancelled";
        case HILUM_ERR_VISION_FAILED:  return "vision processing failed";
    }
    return "unknown error";
}

/* ── Backend info ──────────────────────────────────────────────────────────── */

HILUM_API const char * hilum_backend_info(void) {
    ensure_backend();
    return llama_print_system_info();
}

HILUM_API const char * hilum_backend_version(void) {
    return "hilum-llm v1.0.0 (hilum-local-llm-engine)";
}

HILUM_API uint32_t hilum_api_version(void) {
    return HILUM_API_VERSION;
}

/* ── Cross-thread cancellation ─────────────────────────────────────────────── */

HILUM_API void hilum_cancel(hilum_context * ctx) {
    if (ctx) ctx->cancel_flag.store(true, std::memory_order_release);
}

HILUM_API void hilum_cancel_clear(hilum_context * ctx) {
    if (ctx) ctx->cancel_flag.store(false, std::memory_order_release);
}

/* ── Model ─────────────────────────────────────────────────────────────────── */

HILUM_API hilum_model_params hilum_model_default_params(void) {
    hilum_model_params p;
    p.n_gpu_layers = -1;
    p.use_mmap     = true;
    return p;
}

HILUM_API hilum_error hilum_model_load(
    const char * path, hilum_model_params params, hilum_model ** out_model)
{
    if (!path || !out_model) return HILUM_ERR_INVALID_HANDLE;
    *out_model = nullptr;

    ensure_backend();

    llama_model_params lp = llama_model_default_params();
    if (params.n_gpu_layers >= 0) lp.n_gpu_layers = params.n_gpu_layers;
    lp.use_mmap = params.use_mmap;

    llama_model * llm = llama_model_load_from_file(path, lp);
    if (!llm) {
        set_error("hilum_model_load: failed to load model from '%s'", path);
        return HILUM_ERR_LOAD_FAILED;
    }

    auto * m = new (std::nothrow) hilum_model{llm};
    if (!m) { llama_model_free(llm); return HILUM_ERR_OOM; }

    register_handle(m);
    *out_model = m;
    return HILUM_OK;
}

HILUM_API hilum_error hilum_model_load_from_buffer(
    const void * data, size_t size, hilum_model_params params, hilum_model ** out_model)
{
    if (!data || !out_model) return HILUM_ERR_INVALID_HANDLE;
    *out_model = nullptr;

    ensure_backend();

    llama_model_params lp = llama_model_default_params();
    if (params.n_gpu_layers >= 0) lp.n_gpu_layers = params.n_gpu_layers;
    lp.use_mmap = false;

    llama_model * llm = llama_model_load_from_buffer(data, size, lp);
    if (!llm) {
        set_error("hilum_model_load_from_buffer: failed to load model (%zu bytes)", size);
        return HILUM_ERR_LOAD_FAILED;
    }

    auto * m = new (std::nothrow) hilum_model{llm};
    if (!m) { llama_model_free(llm); return HILUM_ERR_OOM; }

    register_handle(m);
    *out_model = m;
    return HILUM_OK;
}

HILUM_API uint64_t hilum_model_size(const hilum_model * model) {
    if (!model) return 0;
    return llama_model_size(model->llm);
}

HILUM_API void hilum_model_free(hilum_model * model) {
    if (!model) return;
    if (unregister_handle(model)) {
        llama_model_free(model->llm);
        delete model;
    }
}

/* ── Context ───────────────────────────────────────────────────────────────── */

HILUM_API hilum_context_params hilum_context_default_params(void) {
    hilum_context_params p;
    p.n_ctx       = 0;
    p.n_batch     = 0;
    p.n_threads   = 0;
    p.flash_attn  = 0;
    p.type_k      = -1;
    p.type_v      = -1;
    p.n_seq_max   = 0;
    p.draft_model = nullptr;
    p.draft_n_max = 16;
    return p;
}

HILUM_API hilum_error hilum_context_create(
    hilum_model * model, hilum_context_params params, hilum_context ** out_ctx)
{
    if (!model || !out_ctx) return HILUM_ERR_INVALID_HANDLE;
    *out_ctx = nullptr;

    llama_context_params lp = llama_context_default_params();
    if (params.n_ctx > 0)     lp.n_ctx     = params.n_ctx;
    if (params.n_batch > 0)   lp.n_batch   = params.n_batch;
    if (params.n_threads > 0) lp.n_threads = static_cast<int32_t>(params.n_threads);
    if (params.flash_attn)    lp.flash_attn_type = static_cast<llama_flash_attn_type>(params.flash_attn);
    if (params.type_k >= 0)   lp.type_k    = static_cast<ggml_type>(params.type_k);
    if (params.type_v >= 0)   lp.type_v    = static_cast<ggml_type>(params.type_v);
    if (params.n_seq_max > 0) lp.n_seq_max = params.n_seq_max;

    llama_context * llm = llama_init_from_model(model->llm, lp);
    if (!llm) {
        set_error("hilum_context_create: failed to create context (n_ctx=%u)", params.n_ctx);
        return HILUM_ERR_CONTEXT_FAILED;
    }

    auto * c = new (std::nothrow) hilum_context{};
    if (!c) { llama_free(llm); return HILUM_ERR_OOM; }
    c->llm           = llm;
    c->model_llm     = model->llm;
    c->draft_llm     = nullptr;
    c->draft_model_llm = nullptr;
    c->draft_n_max   = 0;
    if (!c) { llama_free(llm); return HILUM_ERR_OOM; }

    if (params.draft_model) {
        llama_context_params dlp = llama_context_default_params();
        dlp.n_ctx     = lp.n_ctx;
        dlp.n_batch   = lp.n_batch;
        if (params.n_threads > 0) dlp.n_threads = static_cast<int32_t>(params.n_threads);

        llama_context * dft = llama_init_from_model(params.draft_model->llm, dlp);
        if (!dft) {
            llama_free(llm);
            delete c;
            return HILUM_ERR_CONTEXT_FAILED;
        }

        c->draft_llm       = dft;
        c->draft_model_llm = params.draft_model->llm;
        c->draft_n_max     = params.draft_n_max > 0 ? params.draft_n_max : 16;
    }

    register_handle(c);
    *out_ctx = c;
    return HILUM_OK;
}

HILUM_API uint32_t hilum_context_size(const hilum_context * ctx) {
    if (!ctx) return 0;
    return static_cast<uint32_t>(llama_n_ctx(ctx->llm));
}

HILUM_API void hilum_context_free(hilum_context * ctx) {
    if (!ctx) return;
    if (unregister_handle(ctx)) {
        evict_sampler_cache(ctx->llm);
        if (ctx->draft_llm) {
            evict_sampler_cache(ctx->draft_llm);
            llama_free(ctx->draft_llm);
        }
        llama_free(ctx->llm);
        delete ctx;
    }
}

HILUM_API void hilum_context_kv_clear(hilum_context * ctx, int32_t from_pos) {
    if (!ctx) return;
    llama_memory_t mem = llama_get_memory(ctx->llm);
    if (mem) llama_memory_seq_rm(mem, 0, from_pos, -1);
}

/* ── Tokenization ──────────────────────────────────────────────────────────── */

HILUM_API int32_t hilum_tokenize(
    const hilum_model * model, const char * text, int32_t text_len,
    int32_t * tokens, int32_t max_tokens,
    bool add_special, bool parse_special)
{
    if (!model || !text) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    return llama_tokenize(vocab, text, text_len, (llama_token *)tokens, max_tokens,
                          add_special, parse_special);
}

HILUM_API int32_t hilum_detokenize(
    const hilum_model * model, const int32_t * tokens, int32_t n_tokens,
    char * buf, int32_t buf_size)
{
    if (!model) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    int32_t n = llama_detokenize(vocab, (const llama_token *)tokens, n_tokens,
                                  buf, buf_size, false, true);
    if (n < 0) {
        int32_t needed = -n;
        if (buf_size >= needed) {
            n = llama_detokenize(vocab, (const llama_token *)tokens, n_tokens,
                                  buf, needed, false, true);
        } else {
            return n; // caller needs bigger buffer (negative)
        }
    }
    return n;
}

HILUM_API int32_t hilum_chat_template(
    const hilum_model * model, const char * messages_json,
    bool add_assistant, char * buf, int32_t buf_size)
{
    if (!model || !messages_json) return 0;

    try {
        auto j = nlohmann::json::parse(messages_json);
        if (!j.is_array()) return 0;

        uint32_t n_msgs = static_cast<uint32_t>(j.size());
        std::vector<std::string> roles(n_msgs), contents(n_msgs);
        std::vector<llama_chat_message> messages(n_msgs);

        for (uint32_t i = 0; i < n_msgs; i++) {
            roles[i]    = j[i].value("role", "");
            contents[i] = j[i].value("content", "");
            messages[i].role    = roles[i].c_str();
            messages[i].content = contents[i].c_str();
        }

        const char * tmpl = llama_model_chat_template(model->llm, nullptr);

        int32_t len = llama_chat_apply_template(
            tmpl, messages.data(), n_msgs, add_assistant, nullptr, 0);
        if (len < 0) return 0;

        if (buf && buf_size > len) {
            llama_chat_apply_template(
                tmpl, messages.data(), n_msgs, add_assistant, buf, buf_size);
            return len;
        }
        return -len; // need bigger buffer
    } catch (...) {
        return 0;
    }
}

/* ── Generation parameters ─────────────────────────────────────────────────── */

HILUM_API hilum_gen_params hilum_gen_default_params(void) {
    hilum_gen_params p;
    p.max_tokens         = 512;
    p.temperature        = 0.7f;
    p.top_p              = 0.9f;
    p.top_k              = 40;
    p.repeat_penalty     = 1.1f;
    p.frequency_penalty  = 0.0f;
    p.presence_penalty   = 0.0f;
    p.seed               = LLAMA_DEFAULT_SEED;
    p.grammar            = nullptr;
    p.grammar_root       = nullptr;
    p.n_past             = 0;
    p.progress_callback  = nullptr;
    p.progress_user_data = nullptr;
    p.stop_sequences     = nullptr;
    p.n_stop_sequences   = 0;
    return p;
}

/* ── Chunked prompt evaluation ─────────────────────────────────────────────── */

static hilum_error eval_prompt_chunked(
    llama_context * ctx,
    const llama_token * tokens, int32_t n_tokens,
    int32_t eval_start,
    hilum_progress_callback progress_cb, void * progress_ud)
{
    int32_t n_batch = llama_n_batch(ctx);
    int32_t eval_count = n_tokens - eval_start;
    if (eval_count <= 0) return HILUM_OK;

    for (int32_t i = eval_start; i < n_tokens; i += n_batch) {
        int32_t chunk_size = std::min(n_batch, n_tokens - i);
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens + i), chunk_size);
        if (llama_decode(ctx, batch) != 0)
            return HILUM_ERR_DECODE_FAILED;
        if (progress_cb) {
            if (!progress_cb(i + chunk_size - eval_start, eval_count, progress_ud))
                return HILUM_ERR_CANCELLED;
        }
    }
    return HILUM_OK;
}

/* ── Speculative decoding helpers ───────────────────────────────────────────── */

/*
 * Draft n tokens greedily from the draft model starting from id_last.
 * Returns the drafted tokens. Stops early if the draft model is uncertain.
 */
static std::vector<llama_token> draft_tokens(
    llama_context * ctx_dft,
    const llama_vocab * vocab_dft,
    llama_token id_last,
    int32_t n_past_dft,
    int32_t n_max)
{
    std::vector<llama_token> result;
    result.reserve(n_max);

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * smpl_dft = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(smpl_dft, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(&id_last, 1);
    batch.pos     = &n_past_dft;
    batch.logits  = nullptr;

    llama_pos pos = n_past_dft;

    // Manually construct single-token batches to control position
    {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0]      = id_last;
        b.pos[0]        = pos++;
        b.n_seq_id[0]   = 1;
        b.seq_id[0][0]  = 0;
        b.logits[0]     = 1;
        b.n_tokens      = 1;

        if (llama_decode(ctx_dft, b) != 0) {
            llama_batch_free(b);
            llama_sampler_free(smpl_dft);
            return result;
        }
        llama_batch_free(b);
    }

    for (int32_t i = 0; i < n_max; i++) {
        llama_token id = llama_sampler_sample(smpl_dft, ctx_dft, -1);
        llama_sampler_accept(smpl_dft, id);
        result.push_back(id);

        if (llama_vocab_is_eog(vocab_dft, id)) break;

        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0]      = id;
        b.pos[0]        = pos++;
        b.n_seq_id[0]   = 1;
        b.seq_id[0][0]  = 0;
        b.logits[0]     = 1;
        b.n_tokens      = 1;

        if (llama_decode(ctx_dft, b) != 0) {
            llama_batch_free(b);
            break;
        }
        llama_batch_free(b);
    }

    llama_sampler_free(smpl_dft);
    return result;
}

/*
 * Verify draft tokens against the target model.
 * Decodes [id_last, draft_0, ..., draft_{n-1}] in a single batch on the target,
 * then samples at each position and compares with the draft.
 * Returns the accepted tokens (always at least 1).
 */
static std::vector<llama_token> verify_draft(
    llama_context * ctx_tgt,
    llama_sampler * smpl,
    llama_token id_last,
    int32_t n_past_tgt,
    const std::vector<llama_token> & draft)
{
    int32_t n_draft = static_cast<int32_t>(draft.size());
    int32_t n_batch_tokens = 1 + n_draft; // id_last + all draft tokens

    llama_batch batch = llama_batch_init(n_batch_tokens, 0, 1);
    batch.token[0]      = id_last;
    batch.pos[0]        = n_past_tgt;
    batch.n_seq_id[0]   = 1;
    batch.seq_id[0][0]  = 0;
    batch.logits[0]     = 1;

    for (int32_t i = 0; i < n_draft; i++) {
        batch.token[1 + i]      = draft[i];
        batch.pos[1 + i]        = n_past_tgt + 1 + i;
        batch.n_seq_id[1 + i]   = 1;
        batch.seq_id[1 + i][0]  = 0;
        batch.logits[1 + i]     = 1;
    }
    batch.n_tokens = n_batch_tokens;

    if (llama_decode(ctx_tgt, batch) != 0) {
        llama_batch_free(batch);
        return {};
    }
    llama_batch_free(batch);

    std::vector<llama_token> accepted;
    accepted.reserve(n_draft + 1);

    for (int32_t i = 0; i < n_draft; i++) {
        llama_token tgt_token = llama_sampler_sample(smpl, ctx_tgt, i);
        llama_sampler_accept(smpl, tgt_token);
        accepted.push_back(tgt_token);

        if (tgt_token != draft[i]) {
            break; // divergence — use target's token, stop
        }
    }

    // If all draft tokens matched, sample one bonus token from the last position
    if (static_cast<int32_t>(accepted.size()) == n_draft) {
        llama_token bonus = llama_sampler_sample(smpl, ctx_tgt, n_draft);
        llama_sampler_accept(smpl, bonus);
        accepted.push_back(bonus);
    }

    return accepted;
}

/*
 * Speculative generation: synchronous, full response into buffer.
 * Uses draft model to predict multiple tokens, verifies in batch on target.
 */
static hilum_error generate_speculative(
    hilum_context * ctx,
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tokens,
    hilum_gen_params params,
    char * buf, int32_t buf_size, int32_t * out_tokens_generated)
{
    int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());
    if (n_prompt == 0) return HILUM_ERR_TOKENIZE_FAILED;

    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    SamplerParams sp = to_sampler_params(params);
    llama_sampler * smpl = create_sampler(
        vocab, sp.top_k, sp.top_p, sp.temperature,
        sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
        sp.seed, grammar_str, grammar_root, ctx->llm);

    // Eval prompt on target (all tokens except last)
    int32_t eval_start = (params.n_past > 0 && params.n_past <= n_prompt - 1) ? params.n_past : 0;
    {
        hilum_error perr = eval_prompt_chunked(
            ctx->llm, prompt_tokens.data(), n_prompt - 1, eval_start,
            params.progress_callback, params.progress_user_data);
        if (perr != HILUM_OK) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return perr;
        }
    }

    // Eval prompt on draft (all tokens except last)
    {
        hilum_error perr = eval_prompt_chunked(
            ctx->draft_llm, prompt_tokens.data(), n_prompt - 1, 0,
            nullptr, nullptr);
        if (perr != HILUM_OK) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return perr;
        }
    }

    llama_token eos = llama_vocab_eos(vocab);
    llama_token id_last = prompt_tokens.back();
    int32_t n_past_tgt = n_prompt - 1;
    int32_t n_past_dft = n_prompt - 1;

    std::string result;
    int32_t generated = 0;

    while (generated < params.max_tokens) {
        // Draft phase
        std::vector<llama_token> draft = draft_tokens(
            ctx->draft_llm, vocab, id_last, n_past_dft, ctx->draft_n_max);

        if (draft.empty()) {
            // Draft failed — fall back to single-token decode on target
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0]      = id_last;
            b.pos[0]        = n_past_tgt;
            b.n_seq_id[0]   = 1;
            b.seq_id[0][0]  = 0;
            b.logits[0]     = 1;
            b.n_tokens      = 1;
            if (llama_decode(ctx->llm, b) != 0) { llama_batch_free(b); break; }
            llama_batch_free(b);

            llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
            llama_sampler_accept(smpl, token);
            if (token == eos || llama_vocab_is_eog(vocab, token)) break;

            token_append(vocab, token, result);
            generated++;
            n_past_tgt++;
            id_last = token;
            continue;
        }

        // Verify phase
        std::vector<llama_token> accepted = verify_draft(
            ctx->llm, smpl, id_last, n_past_tgt, draft);

        if (accepted.empty()) break;

        bool hit_eos = false;
        int32_t n_accepted = 0;

        for (auto token : accepted) {
            if (token == eos || llama_vocab_is_eog(vocab, token)) {
                hit_eos = true;
                break;
            }
            token_append(vocab, token, result);
            n_accepted++;
            generated++;
            if (generated >= params.max_tokens) break;
        }

        // id_last consumed 1 KV position, plus n_accepted generated tokens
        n_past_tgt += 1 + n_accepted;
        n_past_dft += 1 + n_accepted;

        {
            llama_memory_t mem_tgt = llama_get_memory(ctx->llm);
            if (mem_tgt) llama_memory_seq_rm(mem_tgt, 0, n_past_tgt, -1);

            llama_memory_t mem_dft = llama_get_memory(ctx->draft_llm);
            if (mem_dft) llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
        }

        if (hit_eos || generated >= params.max_tokens) break;

        id_last = accepted[n_accepted - 1];
    }

    return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());

    if (buf && buf_size > 0) {
        int32_t copy_len = static_cast<int32_t>(result.size());
        if (copy_len >= buf_size) copy_len = buf_size - 1;
        memcpy(buf, result.data(), copy_len);
        buf[copy_len] = '\0';
    }
    if (out_tokens_generated) *out_tokens_generated = generated;

    return HILUM_OK;
}

/*
 * Speculative generation: streaming, calls callback for each accepted token.
 */
static hilum_error generate_stream_speculative(
    hilum_context * ctx,
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tokens,
    hilum_gen_params params,
    hilum_token_callback callback, void * user_data)
{
    int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());
    if (n_prompt == 0) return HILUM_ERR_TOKENIZE_FAILED;

    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    SamplerParams sp = to_sampler_params(params);
    llama_sampler * smpl = create_sampler(
        vocab, sp.top_k, sp.top_p, sp.temperature,
        sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
        sp.seed, grammar_str, grammar_root, ctx->llm);

    int32_t eval_start = (params.n_past > 0 && params.n_past <= n_prompt - 1) ? params.n_past : 0;
    {
        hilum_error perr = eval_prompt_chunked(
            ctx->llm, prompt_tokens.data(), n_prompt - 1, eval_start,
            params.progress_callback, params.progress_user_data);
        if (perr != HILUM_OK) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return perr;
        }
    }
    {
        hilum_error perr = eval_prompt_chunked(
            ctx->draft_llm, prompt_tokens.data(), n_prompt - 1, 0,
            nullptr, nullptr);
        if (perr != HILUM_OK) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return perr;
        }
    }

    llama_token eos = llama_vocab_eos(vocab);
    llama_token id_last = prompt_tokens.back();
    int32_t n_past_tgt = n_prompt - 1;
    int32_t n_past_dft = n_prompt - 1;
    int32_t generated = 0;

    while (generated < params.max_tokens) {
        std::vector<llama_token> draft = draft_tokens(
            ctx->draft_llm, vocab, id_last, n_past_dft, ctx->draft_n_max);

        if (draft.empty()) {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0]      = id_last;
            b.pos[0]        = n_past_tgt;
            b.n_seq_id[0]   = 1;
            b.seq_id[0][0]  = 0;
            b.logits[0]     = 1;
            b.n_tokens      = 1;
            if (llama_decode(ctx->llm, b) != 0) { llama_batch_free(b); break; }
            llama_batch_free(b);

            llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
            llama_sampler_accept(smpl, token);
            if (token == eos || llama_vocab_is_eog(vocab, token)) break;

            t_token_buf.decode(vocab, token);
            if (!callback(t_token_buf.ptr, t_token_buf.len, user_data)) {
                return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
                return HILUM_ERR_CANCELLED;
            }
            generated++;
            n_past_tgt++;
            id_last = token;
            continue;
        }

        std::vector<llama_token> accepted = verify_draft(
            ctx->llm, smpl, id_last, n_past_tgt, draft);

        if (accepted.empty()) break;

        bool hit_eos = false;
        bool cancelled = false;
        int32_t n_accepted = 0;

        for (auto token : accepted) {
            if (token == eos || llama_vocab_is_eog(vocab, token)) {
                hit_eos = true;
                break;
            }
            t_token_buf.decode(vocab, token);
            if (!callback(t_token_buf.ptr, t_token_buf.len, user_data)) {
                cancelled = true;
                break;
            }
            n_accepted++;
            generated++;
            if (generated >= params.max_tokens) break;
        }

        n_past_tgt += 1 + n_accepted;
        n_past_dft += 1 + n_accepted;

        {
            llama_memory_t mem_tgt = llama_get_memory(ctx->llm);
            if (mem_tgt) llama_memory_seq_rm(mem_tgt, 0, n_past_tgt, -1);

            llama_memory_t mem_dft = llama_get_memory(ctx->draft_llm);
            if (mem_dft) llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
        }

        if (cancelled) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return HILUM_ERR_CANCELLED;
        }
        if (hit_eos || generated >= params.max_tokens) break;

        id_last = accepted[n_accepted - 1];
    }

    return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
    return HILUM_OK;
}

/* ── Tokenize prompt helper ────────────────────────────────────────────────── */

static std::vector<llama_token> tokenize_prompt(
    const llama_vocab * vocab, const char * prompt)
{
    int32_t prompt_len = static_cast<int32_t>(strlen(prompt));
    int32_t n = llama_tokenize(vocab, prompt, prompt_len, nullptr, 0, true, false);
    if (n >= 0) n = 0; else n = -n;
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt, prompt_len, tokens.data(), n, true, false);
    return tokens;
}

/* ── Stop-sequence matching ─────────────────────────────────────────────────── */

/*
 * Returns the index into `result` where a stop sequence match begins,
 * or std::string::npos if none matched.
 */
static size_t check_stop_sequences(
    const std::string & result,
    const char * const * stop_seqs, int32_t n_stop)
{
    if (!stop_seqs || n_stop <= 0) return std::string::npos;
    for (int32_t i = 0; i < n_stop; i++) {
        if (!stop_seqs[i]) continue;
        size_t slen = strlen(stop_seqs[i]);
        if (slen == 0 || result.size() < slen) continue;
        if (result.compare(result.size() - slen, slen, stop_seqs[i]) == 0)
            return result.size() - slen;
    }
    return std::string::npos;
}

/* ── Text inference: synchronous ───────────────────────────────────────────── */

HILUM_API hilum_error hilum_generate(
    hilum_model * model, hilum_context * ctx,
    const char * prompt, hilum_gen_params params,
    char * buf, int32_t buf_size, int32_t * out_tokens_generated)
{
    if (!model || !ctx || !prompt) {
        set_error("hilum_generate: model, ctx, or prompt is NULL");
        return HILUM_ERR_INVALID_HANDLE;
    }

    g_last_error.clear();
    ctx->cancel_flag.store(false, std::memory_order_relaxed);

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    auto prompt_tokens = tokenize_prompt(vocab, prompt);
    int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());

    // Dispatch to speculative path if draft model is available
    if (ctx->draft_llm && n_prompt >= 2) {
        return generate_speculative(ctx, vocab, prompt_tokens, params,
                                    buf, buf_size, out_tokens_generated);
    }

    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    SamplerParams sp = to_sampler_params(params);
    llama_sampler * smpl = create_sampler(
        vocab, sp.top_k, sp.top_p, sp.temperature,
        sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
        sp.seed, grammar_str, grammar_root, ctx->llm);

    // Chunked prompt eval (skip n_past)
    int32_t eval_start = (params.n_past > 0 && params.n_past <= n_prompt) ? params.n_past : 0;

    {
        hilum_error perr = eval_prompt_chunked(
            ctx->llm, prompt_tokens.data(), n_prompt, eval_start,
            params.progress_callback, params.progress_user_data);
        if (perr != HILUM_OK) {
            if (perr == HILUM_ERR_DECODE_FAILED)
                set_error("hilum_generate: prompt eval decode failed at batch");
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return perr;
        }
    }

    llama_token eos = llama_vocab_eos(vocab);
    std::string result;
    int32_t generated = 0;
    hilum_error ret = HILUM_OK;

    for (int32_t i = 0; i < params.max_tokens; i++) {
        if (ctx->cancel_flag.load(std::memory_order_acquire)) {
            ret = HILUM_ERR_CANCELLED;
            break;
        }

        llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
        if (token == eos || llama_vocab_is_eog(vocab, token)) break;

        token_append(vocab, token, result);
        llama_sampler_accept(smpl, token);
        generated++;

        // Check stop sequences
        size_t stop_pos = check_stop_sequences(result, params.stop_sequences, params.n_stop_sequences);
        if (stop_pos != std::string::npos) {
            result.resize(stop_pos);
            break;
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx->llm, next) != 0) {
            set_error("hilum_generate: decode failed at token %d", generated);
            ret = HILUM_ERR_DECODE_FAILED;
            break;
        }
    }

    return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());

    if (buf && buf_size > 0) {
        int32_t copy_len = static_cast<int32_t>(result.size());
        if (copy_len >= buf_size) copy_len = buf_size - 1;
        memcpy(buf, result.data(), copy_len);
        buf[copy_len] = '\0';
    }
    if (out_tokens_generated) *out_tokens_generated = generated;

    return ret;
}

/* ── Text inference: streaming ─────────────────────────────────────────────── */

HILUM_API hilum_error hilum_generate_stream(
    hilum_model * model, hilum_context * ctx,
    const char * prompt, hilum_gen_params params,
    hilum_token_callback callback, void * user_data)
{
    if (!model || !ctx || !prompt || !callback) {
        set_error("hilum_generate_stream: model, ctx, prompt, or callback is NULL");
        return HILUM_ERR_INVALID_HANDLE;
    }

    g_last_error.clear();
    ctx->cancel_flag.store(false, std::memory_order_relaxed);

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    auto prompt_tokens = tokenize_prompt(vocab, prompt);
    int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());

    // Dispatch to speculative path if draft model is available
    if (ctx->draft_llm && n_prompt >= 2) {
        return generate_stream_speculative(ctx, vocab, prompt_tokens, params,
                                           callback, user_data);
    }

    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    SamplerParams sp = to_sampler_params(params);
    llama_sampler * smpl = create_sampler(
        vocab, sp.top_k, sp.top_p, sp.temperature,
        sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
        sp.seed, grammar_str, grammar_root, ctx->llm);

    int32_t eval_start = (params.n_past > 0 && params.n_past <= n_prompt) ? params.n_past : 0;

    {
        hilum_error perr = eval_prompt_chunked(
            ctx->llm, prompt_tokens.data(), n_prompt, eval_start,
            params.progress_callback, params.progress_user_data);
        if (perr != HILUM_OK) {
            if (perr == HILUM_ERR_DECODE_FAILED)
                set_error("hilum_generate_stream: prompt eval decode failed");
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return perr;
        }
    }

    llama_token eos = llama_vocab_eos(vocab);
    std::string accumulated; // for stop-sequence matching
    bool has_stop_seqs = (params.stop_sequences && params.n_stop_sequences > 0);

    for (int32_t i = 0; i < params.max_tokens; i++) {
        if (ctx->cancel_flag.load(std::memory_order_acquire)) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return HILUM_ERR_CANCELLED;
        }

        llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
        if (token == eos || llama_vocab_is_eog(vocab, token)) break;

        t_token_buf.decode(vocab, token);

        // Stop sequence check (accumulate only if needed)
        if (has_stop_seqs) {
            accumulated.append(t_token_buf.ptr, t_token_buf.len);
            if (check_stop_sequences(accumulated, params.stop_sequences,
                                     params.n_stop_sequences) != std::string::npos) {
                break;
            }
        }

        bool keep_going = callback(t_token_buf.ptr, t_token_buf.len, user_data);

        llama_sampler_accept(smpl, token);
        if (!keep_going) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return HILUM_ERR_CANCELLED;
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx->llm, next) != 0) {
            set_error("hilum_generate_stream: decode failed at token %d", i);
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return HILUM_ERR_DECODE_FAILED;
        }
    }

    return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
    return HILUM_OK;
}

/* ── Vision ────────────────────────────────────────────────────────────────── */

HILUM_API hilum_error hilum_mtmd_load(
    hilum_model * model, const char * projector_path,
    hilum_mtmd_params params, hilum_mtmd ** out_mtmd)
{
    if (!model || !projector_path || !out_mtmd) return HILUM_ERR_INVALID_HANDLE;
    *out_mtmd = nullptr;

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = params.use_gpu;
    if (params.n_threads > 0) mp.n_threads = static_cast<int32_t>(params.n_threads);

    mtmd_context * mctx = mtmd_init_from_file(projector_path, model->llm, mp);
    if (!mctx) return HILUM_ERR_VISION_FAILED;

    auto * m = new (std::nothrow) hilum_mtmd{mctx};
    if (!m) { mtmd_free(mctx); return HILUM_ERR_OOM; }

    register_handle(m);
    *out_mtmd = m;
    return HILUM_OK;
}

HILUM_API bool hilum_mtmd_supports_vision(const hilum_mtmd * mtmd) {
    if (!mtmd) return false;
    return mtmd_support_vision(mtmd->ctx);
}

HILUM_API void hilum_mtmd_free(hilum_mtmd * mtmd) {
    if (!mtmd) return;
    if (unregister_handle(mtmd)) {
        mtmd_free(mtmd->ctx);
        delete mtmd;
    }
}

static hilum_error vision_prepare(
    hilum_model * model, hilum_context * ctx, hilum_mtmd * mtmd,
    const char * prompt, const hilum_image * images, int32_t n_images,
    llama_pos * out_n_past)
{
    std::vector<mtmd_bitmap *> bitmaps;
    for (int32_t i = 0; i < n_images; i++) {
        mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_buf(
            mtmd->ctx, images[i].data, images[i].size);
        if (!bmp) {
            for (auto * b : bitmaps) mtmd_bitmap_free(b);
            return HILUM_ERR_VISION_FAILED;
        }
        bitmaps.push_back(bmp);
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text = { prompt, true, false };

    std::vector<const mtmd_bitmap *> bmp_ptrs(bitmaps.size());
    for (size_t i = 0; i < bitmaps.size(); i++) bmp_ptrs[i] = bitmaps[i];

    int32_t tok_res = mtmd_tokenize(mtmd->ctx, chunks, &input_text,
        bmp_ptrs.data(), bmp_ptrs.size());

    for (auto * b : bitmaps) mtmd_bitmap_free(b);

    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        return HILUM_ERR_VISION_FAILED;
    }

    llama_pos n_past = 0;
    int32_t n_batch = llama_n_batch(ctx->llm);
    int32_t eval_res = mtmd_helper_eval_chunks(mtmd->ctx, ctx->llm, chunks,
        n_past, 0, n_batch, true, &n_past);

    mtmd_input_chunks_free(chunks);

    if (eval_res != 0) return HILUM_ERR_DECODE_FAILED;

    *out_n_past = n_past;
    return HILUM_OK;
}

HILUM_API hilum_error hilum_generate_vision(
    hilum_model * model, hilum_context * ctx, hilum_mtmd * mtmd,
    const char * prompt, const hilum_image * images, int32_t n_images,
    hilum_gen_params params, char * buf, int32_t buf_size,
    int32_t * out_tokens_generated)
{
    if (!model || !ctx || !mtmd || !prompt) return HILUM_ERR_INVALID_HANDLE;

    llama_pos n_past = 0;
    hilum_error err = vision_prepare(model, ctx, mtmd, prompt, images, n_images, &n_past);
    if (err != HILUM_OK) return err;

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    SamplerParams sp = to_sampler_params(params);
    llama_sampler * smpl = create_sampler(
        vocab, sp.top_k, sp.top_p, sp.temperature,
        sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
        sp.seed, grammar_str, grammar_root, ctx->llm);

    llama_token eos = llama_vocab_eos(vocab);
    std::string result;
    int32_t generated = 0;

    for (int32_t i = 0; i < params.max_tokens; i++) {
        llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
        if (token == eos) break;

        token_append(vocab, token, result);
        llama_sampler_accept(smpl, token);
        generated++;

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx->llm, next) != 0) break;
    }

    return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());

    if (buf && buf_size > 0) {
        int32_t copy_len = static_cast<int32_t>(result.size());
        if (copy_len >= buf_size) copy_len = buf_size - 1;
        memcpy(buf, result.data(), copy_len);
        buf[copy_len] = '\0';
    }
    if (out_tokens_generated) *out_tokens_generated = generated;

    return HILUM_OK;
}

HILUM_API hilum_error hilum_generate_vision_stream(
    hilum_model * model, hilum_context * ctx, hilum_mtmd * mtmd,
    const char * prompt, const hilum_image * images, int32_t n_images,
    hilum_gen_params params, hilum_token_callback callback, void * user_data)
{
    if (!model || !ctx || !mtmd || !prompt || !callback) return HILUM_ERR_INVALID_HANDLE;

    llama_pos n_past = 0;
    hilum_error err = vision_prepare(model, ctx, mtmd, prompt, images, n_images, &n_past);
    if (err != HILUM_OK) return err;

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    SamplerParams sp = to_sampler_params(params);
    llama_sampler * smpl = create_sampler(
        vocab, sp.top_k, sp.top_p, sp.temperature,
        sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
        sp.seed, grammar_str, grammar_root, ctx->llm);

    llama_token eos = llama_vocab_eos(vocab);

    for (int32_t i = 0; i < params.max_tokens; i++) {
        llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
        if (token == eos) break;

        t_token_buf.decode(vocab, token);
        bool keep_going = callback(t_token_buf.ptr, t_token_buf.len, user_data);

        llama_sampler_accept(smpl, token);
        if (!keep_going) {
            return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
            return HILUM_ERR_CANCELLED;
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx->llm, next) != 0) break;
    }

    return_sampler(ctx->llm, smpl, sp, !grammar_str.empty());
    return HILUM_OK;
}

/* ── Embeddings ────────────────────────────────────────────────────────────── */

HILUM_API hilum_error hilum_emb_context_create(
    hilum_model * model, hilum_emb_params params, hilum_emb_ctx ** out_ctx)
{
    if (!model || !out_ctx) return HILUM_ERR_INVALID_HANDLE;
    *out_ctx = nullptr;

    llama_context_params lp = llama_context_default_params();
    lp.embeddings    = true;
    lp.pooling_type  = LLAMA_POOLING_TYPE_MEAN;
    if (params.n_ctx > 0)     lp.n_ctx     = params.n_ctx;
    if (params.n_batch > 0)   lp.n_batch   = params.n_batch;
    if (params.n_threads > 0) lp.n_threads = static_cast<int32_t>(params.n_threads);
    if (params.pooling_type >= 0) lp.pooling_type = static_cast<enum llama_pooling_type>(params.pooling_type);

    llama_context * llm = llama_init_from_model(model->llm, lp);
    if (!llm) return HILUM_ERR_CONTEXT_FAILED;

    auto * ec = new (std::nothrow) hilum_emb_ctx{llm, model->llm};
    if (!ec) { llama_free(llm); return HILUM_ERR_OOM; }

    register_handle(ec);
    *out_ctx = ec;
    return HILUM_OK;
}

HILUM_API void hilum_emb_context_free(hilum_emb_ctx * ctx) {
    if (!ctx) return;
    if (unregister_handle(ctx)) {
        llama_free(ctx->llm);
        delete ctx;
    }
}

HILUM_API int32_t hilum_emb_dimension(const hilum_model * model) {
    if (!model) return 0;
    return llama_model_n_embd(model->llm);
}

HILUM_API hilum_error hilum_embed(
    hilum_emb_ctx * ctx, hilum_model * model,
    const int32_t * tokens, int32_t n_tokens,
    float * out_embedding, int32_t out_size)
{
    if (!ctx || !model || !tokens || !out_embedding) return HILUM_ERR_INVALID_HANDLE;
    if (n_tokens == 0) return HILUM_OK;

    int32_t n_embd = llama_model_n_embd(model->llm);
    if (out_size < n_embd) return HILUM_ERR_OOM;

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = 1;
    }
    batch.n_tokens = n_tokens;

    llama_memory_t mem = llama_get_memory(ctx->llm);
    if (mem) llama_memory_seq_rm(mem, 0, 0, -1);

    int32_t ret = llama_encode(ctx->llm, batch);
    llama_batch_free(batch);
    if (ret != 0) return HILUM_ERR_ENCODE_FAILED;

    const float * emb = llama_get_embeddings_seq(ctx->llm, 0);
    if (!emb) return HILUM_ERR_ENCODE_FAILED;

    // L2 normalize
    float norm = 0.0f;
    for (int32_t i = 0; i < n_embd; i++) norm += emb[i] * emb[i];
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (int32_t i = 0; i < n_embd; i++) out_embedding[i] = emb[i] / norm;
    } else {
        for (int32_t i = 0; i < n_embd; i++) out_embedding[i] = 0.0f;
    }

    return HILUM_OK;
}

HILUM_API hilum_error hilum_embed_batch(
    hilum_emb_ctx * ctx, hilum_model * model,
    const int32_t ** token_arrays, const int32_t * token_counts,
    int32_t n_inputs, float ** out_embeddings, int32_t emb_dim)
{
    if (!ctx || !model || !token_arrays || !token_counts || !out_embeddings)
        return HILUM_ERR_INVALID_HANDLE;
    if (n_inputs == 0) return HILUM_OK;

    int32_t n_embd = llama_model_n_embd(model->llm);
    if (emb_dim < n_embd) return HILUM_ERR_OOM;

    int32_t total_tokens = 0;
    for (int32_t i = 0; i < n_inputs; i++) total_tokens += token_counts[i];

    llama_batch batch = llama_batch_init(total_tokens, 0, n_inputs);
    int32_t pos_offset = 0;

    for (int32_t seq = 0; seq < n_inputs; seq++) {
        int32_t n_tok = token_counts[seq];
        for (int32_t j = 0; j < n_tok; j++) {
            int32_t idx = pos_offset + j;
            batch.token[idx]     = token_arrays[seq][j];
            batch.pos[idx]       = j;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = static_cast<llama_seq_id>(seq);
            batch.logits[idx]    = 1;
        }
        pos_offset += n_tok;
    }
    batch.n_tokens = total_tokens;

    llama_memory_t mem = llama_get_memory(ctx->llm);
    if (mem) {
        for (int32_t seq = 0; seq < n_inputs; seq++)
            llama_memory_seq_rm(mem, static_cast<llama_seq_id>(seq), 0, -1);
    }

    int32_t ret = llama_encode(ctx->llm, batch);
    llama_batch_free(batch);
    if (ret != 0) return HILUM_ERR_ENCODE_FAILED;

    for (int32_t seq = 0; seq < n_inputs; seq++) {
        const float * emb = llama_get_embeddings_seq(ctx->llm, static_cast<llama_seq_id>(seq));
        if (emb) {
            float norm = 0.0f;
            for (int32_t i = 0; i < n_embd; i++) norm += emb[i] * emb[i];
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                for (int32_t i = 0; i < n_embd; i++) out_embeddings[seq][i] = emb[i] / norm;
            } else {
                for (int32_t i = 0; i < n_embd; i++) out_embeddings[seq][i] = 0.0f;
            }
        } else {
            for (int32_t i = 0; i < n_embd; i++) out_embeddings[seq][i] = 0.0f;
        }
    }

    return HILUM_OK;
}

/* ── Batch inference ───────────────────────────────────────────────────────── */

HILUM_API hilum_error hilum_generate_batch(
    hilum_model * model, hilum_context * ctx,
    const char ** prompts, int32_t n_prompts,
    hilum_gen_params params, hilum_batch_callback callback, void * user_data)
{
    if (!model || !ctx || !prompts || !callback || n_prompts <= 0)
        return HILUM_ERR_INVALID_HANDLE;

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    llama_token eos = llama_vocab_eos(vocab);
    int32_t n_batch_size = llama_n_batch(ctx->llm);

    std::string grammar_str  = params.grammar      ? params.grammar      : "";
    std::string grammar_root = params.grammar_root  ? params.grammar_root  : "root";

    // 1. Tokenize all prompts
    std::vector<std::vector<llama_token>> all_tokens(n_prompts);
    int32_t total_prompt_tokens = 0;
    for (int32_t i = 0; i < n_prompts; i++) {
        all_tokens[i] = tokenize_prompt(vocab, prompts[i]);
        total_prompt_tokens += static_cast<int32_t>(all_tokens[i].size());
    }

    // 2. Create one sampler per sequence
    std::vector<llama_sampler *> samplers(n_prompts);
    for (int32_t i = 0; i < n_prompts; i++) {
        samplers[i] = create_sampler(
            vocab, params.top_k, params.top_p, params.temperature,
            params.repeat_penalty, params.frequency_penalty, params.presence_penalty,
            params.seed, grammar_str, grammar_root, nullptr);
    }

    // 3. Clear KV cache
    llama_memory_t mem = llama_get_memory(ctx->llm);
    if (mem) {
        for (int32_t i = 0; i < n_prompts; i++)
            llama_memory_seq_rm(mem, static_cast<llama_seq_id>(i), 0, -1);
    }

    // 4. Prompt eval with chunking
    {
        struct Entry { llama_token token; llama_pos pos; llama_seq_id seq_id; bool logits; };
        std::vector<Entry> entries;
        entries.reserve(total_prompt_tokens);
        for (int32_t s = 0; s < n_prompts; s++) {
            int32_t n_tok = static_cast<int32_t>(all_tokens[s].size());
            for (int32_t j = 0; j < n_tok; j++) {
                entries.push_back({all_tokens[s][j], j,
                    static_cast<llama_seq_id>(s), j == n_tok - 1});
            }
        }

        for (size_t offset = 0; offset < entries.size(); offset += n_batch_size) {
            size_t chunk_end = std::min(entries.size(), offset + static_cast<size_t>(n_batch_size));
            int32_t chunk_size = static_cast<int32_t>(chunk_end - offset);

            llama_batch batch = llama_batch_init(chunk_size, 0, n_prompts);
            for (int32_t i = 0; i < chunk_size; i++) {
                auto & e = entries[offset + i];
                batch.token[i]     = e.token;
                batch.pos[i]       = e.pos;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = e.seq_id;

                if (e.logits) {
                    bool is_last = true;
                    for (size_t j = offset + i + 1; j < entries.size(); j++) {
                        if (entries[j].seq_id == e.seq_id) { is_last = false; break; }
                    }
                    batch.logits[i] = is_last ? 1 : 0;
                } else {
                    batch.logits[i] = 0;
                }
            }
            batch.n_tokens = chunk_size;

            int32_t ret = llama_decode(ctx->llm, batch);
            llama_batch_free(batch);
            if (ret != 0) {
                for (auto * s : samplers) llama_sampler_free(s);
                return HILUM_ERR_DECODE_FAILED;
            }
        }
    }

    // 5. Sample first token and generation loop
    std::vector<bool> seq_done(n_prompts, false);
    std::vector<int32_t> seq_generated(n_prompts, 0);
    int32_t n_active = n_prompts;

    // First token for each sequence
    for (int32_t s = 0; s < n_prompts; s++) {
        llama_token token = llama_sampler_sample(samplers[s], ctx->llm, -1);
        if (token == eos) {
            seq_done[s] = true;
            n_active--;
            hilum_batch_event ev{s, nullptr, 0, true, "stop"};
            callback(ev, user_data);
            continue;
        }

        t_token_buf.decode(vocab, token);
        hilum_batch_event ev{s, t_token_buf.ptr, t_token_buf.len, false, nullptr};
        bool keep = callback(ev, user_data);

        llama_sampler_accept(samplers[s], token);
        seq_generated[s]++;
        all_tokens[s].push_back(token);

        if (!keep) {
            seq_done[s] = true;
            n_active--;
            hilum_batch_event done_ev{s, nullptr, 0, true, "cancelled"};
            callback(done_ev, user_data);
        }
    }

    // Generation loop
    while (n_active > 0) {
        int32_t batch_count = 0;
        std::vector<int32_t> active_seqs;
        for (int32_t s = 0; s < n_prompts; s++) {
            if (!seq_done[s]) { active_seqs.push_back(s); batch_count++; }
        }

        llama_batch batch = llama_batch_init(batch_count, 0, n_prompts);
        for (int32_t i = 0; i < batch_count; i++) {
            int32_t s = active_seqs[i];
            llama_token last_token = all_tokens[s].back();
            int32_t pos = static_cast<int32_t>(all_tokens[s].size()) - 1;
            batch.token[i]     = last_token;
            batch.pos[i]       = pos;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = static_cast<llama_seq_id>(s);
            batch.logits[i]    = 1;
        }
        batch.n_tokens = batch_count;

        int32_t ret = llama_decode(ctx->llm, batch);
        llama_batch_free(batch);
        if (ret != 0) break;

        for (int32_t i = 0; i < batch_count; i++) {
            int32_t s = active_seqs[i];
            llama_token token = llama_sampler_sample(samplers[s], ctx->llm, i);

            if (token == eos) {
                seq_done[s] = true;
                n_active--;
                hilum_batch_event ev{s, nullptr, 0, true, "stop"};
                callback(ev, user_data);
                continue;
            }

            seq_generated[s]++;
            if (seq_generated[s] >= params.max_tokens) {
                t_token_buf.decode(vocab, token);
                hilum_batch_event tok_ev{s, t_token_buf.ptr, t_token_buf.len, false, nullptr};
                callback(tok_ev, user_data);
                seq_done[s] = true;
                n_active--;
                hilum_batch_event done_ev{s, nullptr, 0, true, "length"};
                callback(done_ev, user_data);
                continue;
            }

            t_token_buf.decode(vocab, token);
            hilum_batch_event ev{s, t_token_buf.ptr, t_token_buf.len, false, nullptr};
            bool keep = callback(ev, user_data);

            llama_sampler_accept(samplers[s], token);
            all_tokens[s].push_back(token);

            if (!keep) {
                seq_done[s] = true;
                n_active--;
                hilum_batch_event done_ev{s, nullptr, 0, true, "cancelled"};
                callback(done_ev, user_data);
            }
        }
    }

    for (auto * s : samplers) llama_sampler_free(s);
    return HILUM_OK;
}

/* ── Grammar ───────────────────────────────────────────────────────────────── */

HILUM_API int32_t hilum_json_schema_to_grammar(
    const char * schema_json, char * buf, int32_t buf_size)
{
    if (!schema_json) return 0;
    try {
        auto schema = nlohmann::ordered_json::parse(schema_json);
        std::string grammar = json_schema_to_grammar(schema);
        int32_t len = static_cast<int32_t>(grammar.size());
        if (buf && buf_size > len) {
            memcpy(buf, grammar.data(), len);
            buf[len] = '\0';
            return len;
        }
        return -(len + 1); // negative = required buffer size
    } catch (...) {
        return 0;
    }
}

/* ── Quantization ──────────────────────────────────────────────────────────── */

HILUM_API hilum_quantize_params hilum_quantize_default_params(void) {
    hilum_quantize_params p;
    p.ftype                  = 15; // Q4_K_M
    p.nthread                = 0;
    p.allow_requantize       = false;
    p.quantize_output_tensor = true;
    p.pure                   = false;
    return p;
}

HILUM_API hilum_error hilum_quantize(
    const char * input_path, const char * output_path, hilum_quantize_params params)
{
    if (!input_path || !output_path) return HILUM_ERR_INVALID_HANDLE;

    ensure_backend();

    llama_model_quantize_params lp = llama_model_quantize_default_params();
    lp.ftype                  = static_cast<llama_ftype>(params.ftype);
    lp.nthread                = params.nthread;
    lp.allow_requantize       = params.allow_requantize;
    lp.quantize_output_tensor = params.quantize_output_tensor;
    lp.pure                   = params.pure;

    int result = llama_model_quantize(input_path, output_path, &lp);
    return result == 0 ? HILUM_OK : HILUM_ERR_QUANTIZE_FAILED;
}

/* ── Logging ───────────────────────────────────────────────────────────────── */

HILUM_API void hilum_log_set(hilum_log_callback callback, void * user_data) {
    if (g_log_cb) {
        llama_log_set(nullptr, nullptr);
    }
    g_log_cb = callback;
    g_log_ud = user_data;
    if (callback) {
        llama_log_set(native_log_callback, nullptr);
    }
}

HILUM_API void hilum_log_set_level(hilum_log_level min_level) {
    g_log_min_level.store(static_cast<int>(min_level), std::memory_order_relaxed);
}

/* ── Warmup ────────────────────────────────────────────────────────────────── */

HILUM_API hilum_error hilum_warmup(hilum_model * model, hilum_context * ctx) {
    if (!model || !ctx) return HILUM_ERR_INVALID_HANDLE;

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    llama_token bos = llama_vocab_bos(vocab);

    llama_batch batch = llama_batch_get_one(&bos, 1);
    int32_t ret = llama_decode(ctx->llm, batch);

    llama_memory_t mem = llama_get_memory(ctx->llm);
    if (mem) llama_memory_seq_rm(mem, 0, 0, -1);

    return ret == 0 ? HILUM_OK : HILUM_ERR_DECODE_FAILED;
}

/* ── Performance metrics ───────────────────────────────────────────────────── */

HILUM_API hilum_perf_data hilum_get_perf(const hilum_context * ctx) {
    hilum_perf_data result = {};
    if (!ctx) return result;

    struct llama_perf_context_data perf = llama_perf_context(ctx->llm);

    result.prompt_eval_ms  = perf.t_p_eval_ms;
    result.generation_ms   = perf.t_eval_ms;
    result.prompt_tokens   = perf.n_p_eval;
    result.generated_tokens = perf.n_eval;
    result.prompt_tokens_per_sec    = (perf.t_p_eval_ms > 0) ? (perf.n_p_eval * 1000.0 / perf.t_p_eval_ms) : 0.0;
    result.generated_tokens_per_sec = (perf.t_eval_ms > 0)   ? (perf.n_eval   * 1000.0 / perf.t_eval_ms)   : 0.0;

    return result;
}

/* ── Optimal thread count ──────────────────────────────────────────────────── */

#ifdef __APPLE__
static int sysctl_int(const char * name) {
    int val = 0;
    size_t len = sizeof(val);
    if (sysctlbyname(name, &val, &len, nullptr, 0) == 0) return val;
    return 0;
}
#endif

HILUM_API int32_t hilum_optimal_thread_count(void) {
#ifdef __APPLE__
    // Apple Silicon: use P-core (performance level 0) physical count
    int pcores = sysctl_int("hw.perflevel0.physicalcpu");
    if (pcores > 0) {
        int lpm = sysctl_int("kern.lowpowermode");
        if (lpm) return std::max(1, pcores / 2);
        return pcores;
    }
    // Intel Mac: physical CPU count (avoids counting hyperthreads)
    int phys = sysctl_int("hw.physicalcpu");
    if (phys > 0) return std::max(1, phys);
#endif

#ifdef __linux__
    // Detect big.LITTLE by comparing max frequencies across CPUs
    int total_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (total_cores <= 0) total_cores = 4;

    int max_freq = 0;
    int big_cores = 0;
    for (int i = 0; i < total_cores; i++) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                           "/cpufreq/cpuinfo_max_freq";
        std::ifstream f(path);
        int freq = 0;
        if (f >> freq) {
            if (freq > max_freq) {
                max_freq = freq;
                big_cores = 1;
            } else if (freq == max_freq) {
                big_cores++;
            }
        }
    }
    int count = (big_cores > 0 && big_cores < total_cores) ? big_cores
                                                            : std::max(1, total_cores / 2);

    // Check battery-saver / discharging state
    {
        std::ifstream status("/sys/class/power_supply/BAT0/status");
        if (status.good()) {
            std::string state;
            std::getline(status, state);
            if (state == "Discharging") count = std::max(1, count / 2);
        }
    }
    return count;
#endif

    // Fallback: half of logical cores (assumes hyperthreading)
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, hw > 0 ? hw / 2 : 4);
}

/* ── Benchmark ─────────────────────────────────────────────────────────────── */

HILUM_API hilum_benchmark_params hilum_benchmark_default_params(void) {
    hilum_benchmark_params p;
    p.prompt_tokens   = 128;
    p.generate_tokens = 64;
    p.iterations      = 3;
    return p;
}

HILUM_API hilum_error hilum_benchmark(
    hilum_model           * model,
    hilum_context         * ctx,
    hilum_benchmark_params  params,
    hilum_benchmark_result* out)
{
    if (!model || !ctx || !out) return HILUM_ERR_INVALID_HANDLE;

    const llama_vocab * vocab = llama_model_get_vocab(model->llm);
    llama_token bos = llama_vocab_bos(vocab);

    // Build a dummy prompt of the requested length by repeating BOS
    int32_t n_prompt = params.prompt_tokens > 0 ? params.prompt_tokens : 128;
    int32_t n_gen    = params.generate_tokens > 0 ? params.generate_tokens : 64;
    int32_t n_iter   = params.iterations > 0 ? params.iterations : 3;

    std::vector<llama_token> prompt_tokens(n_prompt, bos);

    double sum_pp_tps = 0.0;
    double sum_tg_tps = 0.0;
    double sum_ttft   = 0.0;
    int32_t completed = 0;

    auto wall_start = std::chrono::steady_clock::now();

    for (int32_t iter = 0; iter < n_iter; iter++) {
        // Clear KV cache
        llama_memory_t mem = llama_get_memory(ctx->llm);
        if (mem) llama_memory_seq_rm(mem, 0, 0, -1);

        // Reset perf counters
        llama_perf_context_reset(ctx->llm);

        // Prompt eval (chunked)
        hilum_error perr = eval_prompt_chunked(
            ctx->llm, prompt_tokens.data(), n_prompt, 0, nullptr, nullptr);
        if (perr != HILUM_OK) return perr;

        // Token generation
        llama_token eos = llama_vocab_eos(vocab);
        SamplerParams sp;
        sp.temperature = 0.0f;
        sp.top_p = 1.0f;
        sp.top_k = 1;
        sp.repeat_penalty = 1.0f;
        sp.frequency_penalty = 0.0f;
        sp.presence_penalty = 0.0f;
        sp.seed = 42;

        llama_sampler * smpl = create_sampler(
            vocab, sp.top_k, sp.top_p, sp.temperature,
            sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
            sp.seed, "", "", ctx->llm);

        for (int32_t i = 0; i < n_gen; i++) {
            llama_token token = llama_sampler_sample(smpl, ctx->llm, -1);
            if (token == eos) break;
            llama_sampler_accept(smpl, token);
            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx->llm, next) != 0) break;
        }

        return_sampler(ctx->llm, smpl, sp, false);

        // Collect perf
        struct llama_perf_context_data perf = llama_perf_context(ctx->llm);
        double pp_tps = (perf.t_p_eval_ms > 0) ? (perf.n_p_eval * 1000.0 / perf.t_p_eval_ms) : 0.0;
        double tg_tps = (perf.t_eval_ms > 0)   ? (perf.n_eval   * 1000.0 / perf.t_eval_ms)   : 0.0;

        sum_pp_tps += pp_tps;
        sum_tg_tps += tg_tps;
        sum_ttft   += perf.t_p_eval_ms;
        completed++;
    }

    auto wall_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    double n = completed > 0 ? completed : 1;
    out->prompt_tokens_per_sec    = sum_pp_tps / n;
    out->generated_tokens_per_sec = sum_tg_tps / n;
    out->ttft_ms                  = sum_ttft / n;
    out->total_ms                 = total_ms;
    out->iterations               = completed;

    return HILUM_OK;
}
