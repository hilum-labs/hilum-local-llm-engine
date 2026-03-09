/*
 * hilum_llm.h — Public C API for the Hilum LLM inference engine.
 *
 * This is the stable contract used by every platform binding (Node.js N-API,
 * React Native Turbo Module, Swift, JNI, Dart FFI, Python ctypes, Rust FFI).
 * It wraps the internal llama.cpp / ggml APIs behind opaque handles and a
 * pure C interface so any language with a C FFI can call it.
 *
 * ABI rules:
 *   - New functions may be added freely (non-breaking).
 *   - Existing function signatures never change.
 *   - Use *_default_params() to get structs with sensible defaults;
 *     new fields are always appended and default to zero/false.
 *   - All handles are opaque pointers; internal layout may change.
 */

#ifndef HILUM_LLM_H
#define HILUM_LLM_H

/*
 * API/ABI versioning
 *
 * ABI contract:
 * - Within a major version, changes are additive-only.
 * - Existing exported signatures and struct field ordering are stable.
 * - New struct fields are appended.
 *
 * Runtime wrappers should compare compiled major vs runtime major via
 * hilum_api_version().
 */
#define HILUM_API_VERSION_MAJOR 1
#define HILUM_API_VERSION_MINOR 0
#define HILUM_API_VERSION_PATCH 0
#define HILUM_API_VERSION \
    ((uint32_t)(((HILUM_API_VERSION_MAJOR) << 16) | ((HILUM_API_VERSION_MINOR) << 8) | (HILUM_API_VERSION_PATCH)))

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
#  ifdef HILUM_BUILD
#    define HILUM_API __declspec(dllexport)
#  else
#    define HILUM_API __declspec(dllimport)
#  endif
#else
#  define HILUM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Thread safety contract:
 *
 * - A hilum_model is immutable after load and may be shared across threads.
 *   Multiple contexts may use the same model concurrently.
 *
 * - A hilum_context / hilum_emb_ctx is NOT thread-safe. Use one context per
 *   thread and do not concurrently call generate/embed APIs on the same handle.
 *
 * - hilum_cancel() / hilum_cancel_clear() are thread-safe and may be called
 *   from any thread while another thread is generating on that context.
 *
 * - Global query helpers (hilum_api_version, hilum_backend_info,
 *   hilum_backend_version, hilum_last_error, hilum_error_str) are thread-safe.
 */

/* ── Opaque handles ────────────────────────────────────────────────────────── */

typedef struct hilum_model    hilum_model;
typedef struct hilum_context  hilum_context;
typedef struct hilum_mtmd     hilum_mtmd;
typedef struct hilum_emb_ctx  hilum_emb_ctx;

/* ── Error codes ───────────────────────────────────────────────────────────── */

typedef enum {
    HILUM_OK = 0,
    HILUM_ERR_LOAD_FAILED,
    HILUM_ERR_CONTEXT_FAILED,
    HILUM_ERR_TOKENIZE_FAILED,
    HILUM_ERR_DECODE_FAILED,
    HILUM_ERR_ENCODE_FAILED,
    HILUM_ERR_INVALID_HANDLE,
    HILUM_ERR_OOM,
    HILUM_ERR_GRAMMAR_FAILED,
    HILUM_ERR_QUANTIZE_FAILED,
    HILUM_ERR_CANCELLED,
    HILUM_ERR_VISION_FAILED,
} hilum_error;

HILUM_API const char * hilum_error_str(hilum_error err);

/*
 * Returns a thread-local string with details about the last error.
 * Valid until the next hilum_* call on the same thread. May be empty.
 */
HILUM_API const char * hilum_last_error(void);

/* ── Backend info ──────────────────────────────────────────────────────────── */

HILUM_API const char * hilum_backend_info(void);
HILUM_API const char * hilum_backend_version(void);
HILUM_API uint32_t     hilum_api_version(void);

/* ── Model lifecycle ───────────────────────────────────────────────────────── */

typedef struct {
    int32_t  n_gpu_layers;   /* -1 = auto (offload all layers to GPU) */
    bool     use_mmap;       /* default: true */
} hilum_model_params;

HILUM_API hilum_model_params hilum_model_default_params(void);

HILUM_API hilum_error hilum_model_load(
    const char           * path,
    hilum_model_params     params,
    hilum_model         ** out_model);

HILUM_API hilum_error hilum_model_load_from_buffer(
    const void           * data,
    size_t                 size,
    hilum_model_params     params,
    hilum_model         ** out_model);

HILUM_API uint64_t hilum_model_size(const hilum_model * model);
HILUM_API void     hilum_model_free(hilum_model * model);

/* ── Context lifecycle ─────────────────────────────────────────────────────── */

typedef struct {
    uint32_t n_ctx;          /* 0 = model default */
    uint32_t n_batch;        /* 0 = default 2048 */
    uint32_t n_threads;      /* 0 = auto-detect */
    int32_t  flash_attn;     /* 0 = disabled, 1 = enabled */
    int32_t  type_k;         /* -1 = default (F16) */
    int32_t  type_v;         /* -1 = default (F16) */
    uint32_t n_seq_max;      /* max parallel sequences (default 1) */

    /*
     * Speculative decoding — set draft_model to enable.
     * Uses a small "draft" model to predict multiple tokens, then verifies
     * against the target model in a single batch. 2-3x generation speedup.
     * Requires compatible vocabularies (same model family / tokenizer).
     */
    hilum_model * draft_model;    /* NULL = disabled (default) */
    int32_t       draft_n_max;    /* max draft tokens per step (default: 16) */
} hilum_context_params;

HILUM_API hilum_context_params hilum_context_default_params(void);

HILUM_API hilum_error hilum_context_create(
    hilum_model          * model,
    hilum_context_params   params,
    hilum_context       ** out_ctx);

HILUM_API uint32_t hilum_context_size(const hilum_context * ctx);
HILUM_API void     hilum_context_free(hilum_context * ctx);
HILUM_API void     hilum_context_kv_clear(hilum_context * ctx, int32_t from_pos);

/* ── Cross-thread cancellation ────────────────────────────────────────────── */

/*
 * Thread-safe: can be called from any thread to cancel ongoing inference
 * on the given context. The next token-sampling iteration will return
 * HILUM_ERR_CANCELLED. Call hilum_cancel_clear() before reusing the context.
 */
HILUM_API void hilum_cancel(hilum_context * ctx);
HILUM_API void hilum_cancel_clear(hilum_context * ctx);

/* ── Tokenization ──────────────────────────────────────────────────────────── */

/*
 * Returns number of tokens written. If buffer too small, returns the
 * negative of the required size (-n means n tokens needed).
 */
HILUM_API int32_t hilum_tokenize(
    const hilum_model * model,
    const char        * text,
    int32_t             text_len,
    int32_t           * tokens,
    int32_t             max_tokens,
    bool                add_special,
    bool                parse_special);

HILUM_API int32_t hilum_detokenize(
    const hilum_model * model,
    const int32_t     * tokens,
    int32_t             n_tokens,
    char              * buf,
    int32_t             buf_size);

/*
 * Apply the model's chat template to a JSON array of {role, content} messages.
 * Returns bytes written. If buf is too small, returns the negative of the
 * required size.
 */
HILUM_API int32_t hilum_chat_template(
    const hilum_model * model,
    const char        * messages_json,
    bool                add_assistant,
    char              * buf,
    int32_t             buf_size);

/* ── Progress / abort callback for prompt evaluation ───────────────────────── */

/*
 * Called between prompt-eval chunks. Return false to abort.
 * tokens_processed / tokens_total give prompt-eval progress.
 */
typedef bool (*hilum_progress_callback)(
    int32_t  tokens_processed,
    int32_t  tokens_total,
    void   * user_data);

/* ── Generation parameters ─────────────────────────────────────────────────── */

typedef struct {
    int32_t      max_tokens;          /* default: 512 */
    float        temperature;         /* default: 0.7 */
    float        top_p;               /* default: 0.9 */
    int32_t      top_k;               /* default: 40 */
    float        repeat_penalty;      /* default: 1.1 */
    float        frequency_penalty;   /* default: 0.0 */
    float        presence_penalty;    /* default: 0.0 */
    uint32_t     seed;                /* UINT32_MAX = random */
    const char * grammar;             /* GBNF grammar string (NULL = none) */
    const char * grammar_root;        /* grammar root rule (NULL = "root") */
    int32_t      n_past;              /* tokens already in KV cache (0 = none) */
    hilum_progress_callback progress_callback;  /* NULL = no progress (default) */
    void       * progress_user_data;
    const char * const * stop_sequences;  /* NULL-terminated array, or NULL */
    int32_t      n_stop_sequences;        /* 0 = none */
} hilum_gen_params;

HILUM_API hilum_gen_params hilum_gen_default_params(void);

/* ── Text inference ────────────────────────────────────────────────────────── */

/* Synchronous: generate full response into buf. Returns bytes written. */
HILUM_API hilum_error hilum_generate(
    hilum_model   * model,
    hilum_context * ctx,
    const char    * prompt,
    hilum_gen_params params,
    char          * buf,
    int32_t         buf_size,
    int32_t       * out_tokens_generated);

/*
 * Streaming: calls callback for each token.
 * Return false from callback to cancel generation.
 */
typedef bool (*hilum_token_callback)(
    const char * token,
    int32_t      token_len,
    void       * user_data);

HILUM_API hilum_error hilum_generate_stream(
    hilum_model        * model,
    hilum_context      * ctx,
    const char         * prompt,
    hilum_gen_params     params,
    hilum_token_callback callback,
    void               * user_data);

/* ── Vision / Multimodal ───────────────────────────────────────────────────── */

typedef struct {
    bool     use_gpu;
    uint32_t n_threads;   /* 0 = auto */
} hilum_mtmd_params;

HILUM_API hilum_error hilum_mtmd_load(
    hilum_model      * model,
    const char       * projector_path,
    hilum_mtmd_params  params,
    hilum_mtmd      ** out_mtmd);

HILUM_API bool hilum_mtmd_supports_vision(const hilum_mtmd * mtmd);
HILUM_API void hilum_mtmd_free(hilum_mtmd * mtmd);

typedef struct {
    const uint8_t * data;
    size_t          size;
} hilum_image;

HILUM_API hilum_error hilum_generate_vision(
    hilum_model      * model,
    hilum_context    * ctx,
    hilum_mtmd       * mtmd,
    const char       * prompt,
    const hilum_image* images,
    int32_t            n_images,
    hilum_gen_params   params,
    char             * buf,
    int32_t            buf_size,
    int32_t          * out_tokens_generated);

HILUM_API hilum_error hilum_generate_vision_stream(
    hilum_model        * model,
    hilum_context      * ctx,
    hilum_mtmd         * mtmd,
    const char         * prompt,
    const hilum_image  * images,
    int32_t              n_images,
    hilum_gen_params     params,
    hilum_token_callback callback,
    void               * user_data);

/* ── Embeddings ────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    int32_t  pooling_type;   /* -1 = model default */
} hilum_emb_params;

HILUM_API hilum_error hilum_emb_context_create(
    hilum_model     * model,
    hilum_emb_params  params,
    hilum_emb_ctx  ** out_ctx);

HILUM_API void    hilum_emb_context_free(hilum_emb_ctx * ctx);
HILUM_API int32_t hilum_emb_dimension(const hilum_model * model);

/* Embed a single token sequence. Output is L2-normalized. */
HILUM_API hilum_error hilum_embed(
    hilum_emb_ctx   * ctx,
    hilum_model     * model,
    const int32_t   * tokens,
    int32_t           n_tokens,
    float           * out_embedding,
    int32_t           out_size);

/* Batch embed: multiple inputs, one embedding per input. */
HILUM_API hilum_error hilum_embed_batch(
    hilum_emb_ctx   * ctx,
    hilum_model     * model,
    const int32_t  ** token_arrays,
    const int32_t   * token_counts,
    int32_t           n_inputs,
    float          ** out_embeddings,
    int32_t           emb_dim);

/* ── Batch inference ───────────────────────────────────────────────────────── */

typedef struct {
    int32_t      seq_index;
    const char * token;          /* NULL when done */
    int32_t      token_len;
    bool         done;
    const char * finish_reason;  /* "stop", "length", "cancelled", or NULL */
} hilum_batch_event;

typedef bool (*hilum_batch_callback)(
    hilum_batch_event event,
    void            * user_data);

HILUM_API hilum_error hilum_generate_batch(
    hilum_model        * model,
    hilum_context      * ctx,
    const char        ** prompts,
    int32_t              n_prompts,
    hilum_gen_params     params,
    hilum_batch_callback callback,
    void               * user_data);

/* ── Grammar ───────────────────────────────────────────────────────────────── */

/*
 * Convert JSON Schema string to GBNF grammar string.
 * Returns bytes written. If buf too small, returns the negative of the
 * required size.
 */
HILUM_API int32_t hilum_json_schema_to_grammar(
    const char * schema_json,
    char       * buf,
    int32_t      buf_size);

/* ── Quantization ──────────────────────────────────────────────────────────── */

typedef struct {
    int32_t  ftype;
    int32_t  nthread;              /* 0 = auto */
    bool     allow_requantize;
    bool     quantize_output_tensor;
    bool     pure;
} hilum_quantize_params;

HILUM_API hilum_quantize_params hilum_quantize_default_params(void);

HILUM_API hilum_error hilum_quantize(
    const char           * input_path,
    const char           * output_path,
    hilum_quantize_params  params);

/* ── Logging ───────────────────────────────────────────────────────────────── */

typedef enum {
    HILUM_LOG_DEBUG = 1,
    HILUM_LOG_INFO  = 2,
    HILUM_LOG_WARN  = 3,
    HILUM_LOG_ERROR = 4,
} hilum_log_level;

typedef void (*hilum_log_callback)(
    hilum_log_level  level,
    const char     * text,
    void           * user_data);

HILUM_API void hilum_log_set(hilum_log_callback callback, void * user_data);
HILUM_API void hilum_log_set_level(hilum_log_level min_level);

/* ── Warmup ────────────────────────────────────────────────────────────────── */

HILUM_API hilum_error hilum_warmup(hilum_model * model, hilum_context * ctx);

/* ── Performance metrics ───────────────────────────────────────────────────── */

typedef struct {
    double  prompt_eval_ms;
    double  generation_ms;
    int32_t prompt_tokens;
    int32_t generated_tokens;
    double  prompt_tokens_per_sec;
    double  generated_tokens_per_sec;
} hilum_perf_data;

HILUM_API hilum_perf_data hilum_get_perf(const hilum_context * ctx);

/* ── Optimal thread count ──────────────────────────────────────────────────── */

/*
 * Returns the recommended thread count for LLM inference on this machine.
 * Accounts for P-core / E-core topology, hyperthreading, and low-power mode.
 * Returns at least 1.
 */
HILUM_API int32_t hilum_optimal_thread_count(void);

/* ── Benchmark ─────────────────────────────────────────────────────────────── */

typedef struct {
    int32_t prompt_tokens;    /* tokens to evaluate per iteration (default: 128) */
    int32_t generate_tokens;  /* tokens to generate per iteration (default: 64) */
    int32_t iterations;       /* number of iterations (default: 3) */
} hilum_benchmark_params;

typedef struct {
    double  prompt_tokens_per_sec;
    double  generated_tokens_per_sec;
    double  ttft_ms;        /* averaged time-to-first-token */
    double  total_ms;       /* wall time for all iterations */
    int32_t iterations;     /* iterations actually completed */
} hilum_benchmark_result;

HILUM_API hilum_benchmark_params hilum_benchmark_default_params(void);

HILUM_API hilum_error hilum_benchmark(
    hilum_model           * model,
    hilum_context         * ctx,
    hilum_benchmark_params  params,
    hilum_benchmark_result* out);

#ifdef __cplusplus
}
#endif

#endif /* HILUM_LLM_H */
