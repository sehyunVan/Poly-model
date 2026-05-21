import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY      = os.getenv("GROK_API_KEY")
QWEN_API_KEY      = os.getenv("QWEN_API_KEY")

# ── Model Registry ────────────────────────────────────────────────────────────
# Each entry: name, provider, model_id, api_key
# provider "openai-compat" = OpenAI-compatible REST (DeepSeek, Grok, Qwen all use this)
ALL_MODELS = [
    # claude-opus disabled 2026-04-12 — ~6,200 Anthropic calls/day; opus is 18× more expensive
    # than haiku for a 3-line classification task. Replace with haiku to cut Anthropic spend.
    # Re-enable if haiku vote quality measurably degrades NO WR below 80%.
    # {
    #     "name": "claude-opus",
    #     "provider": "anthropic",
    #     "model": "claude-opus-4-6",
    #     "api_key": ANTHROPIC_API_KEY,
    # },
    # claude-haiku disabled 2026-05-15 — 79% abstain rate (1019 abstain / 273 voted);
    # when it voted, lowest accuracy in the panel (61.5%). Dead weight burning Anthropic credits.
    # {
    #     "name": "claude-haiku",
    #     "provider": "anthropic",
    #     "model": "claude-haiku-4-5-20251001",
    #     "api_key": ANTHROPIC_API_KEY,
    # },
    # claude-sonnet disabled 2026-05-15 — user decision to drop Anthropic to cut API costs.
    # Per paper data sonnet had the highest accuracy (73.3% on 913 votes) — re-enable if
    # Sonnet-solo or AND-gate strategy is revisited and budget allows.
    # {
    #     "name": "claude-sonnet",
    #     "provider": "anthropic",
    #     "model": "claude-sonnet-4-6",
    #     "api_key": ANTHROPIC_API_KEY,
    # },
    # gpt-4o disabled 2026-04-20 — lowest panel accuracy (60.7% all-time, 66.2% NO-only),
    # weakest abstain quality (19% abstain-on-loss vs 37-38% for deepseek/sonnet).
    # Diluted agree_frac signal without adding directional edge. Drop confirmed by
    # pairwise audit: 285 votes, least selective model in the panel.
    # {
    #     "name": "gpt-4o",
    #     "provider": "openai-compat",
    #     "model": "gpt-4o",
    #     "api_key": OPENAI_API_KEY,
    #     "base_url": "https://api.openai.com/v1",
    # },
    # gpt-4o-mini disabled -- redundant with gpt-4o and causes rate limit hits
    # {
    #     "name": "gpt-4o-mini",
    #     "provider": "openai-compat",
    #     "model": "gpt-4o-mini",
    #     "api_key": OPENAI_API_KEY,
    #     "base_url": "https://api.openai.com/v1",
    # },
    {
        "name": "deepseek",
        "provider": "openai-compat",
        "model": "deepseek-chat",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com/v1",
    },
    # grok disabled 2026-04-12 — credits exhausted (429 on every call, wastes retries)
    # Re-enable when credits are topped up:
    # {
    #     "name": "grok",
    #     "provider": "openai-compat",
    #     "model": "grok-3",
    #     "api_key": GROK_API_KEY,
    #     "base_url": "https://api.x.ai/v1",
    # },
    {
        "name": "qwen",
        "provider": "openai-compat",
        "model": "qwen-plus",
        "api_key": QWEN_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
]

# Only include models whose API key is set
ACTIVE_MODELS = [m for m in ALL_MODELS if m.get("api_key")]

# ── Market Filters ────────────────────────────────────────────────────────────
FILTER_MIN_YES_PRICE    = 0.05   # ignore near-certain NO
FILTER_MAX_YES_PRICE    = 0.95   # ignore near-certain YES
FILTER_MIN_VOLUME_24H   = 500    # ignore thin markets ($)
FILTER_MIN_HOURS        = 1.0    # ignore closing soon
FILTER_MAX_HOURS        = 168.0  # ignore closing too far out (7 days)

# Sports tags removed — RAG now provides real-time context for evaluation.
# Only crypto price-action markets are excluded (handled by dedicated crypto loop).
FILTER_EXCLUDED_TAGS = {
    "crypto", "cryptocurrency", "bitcoin", "ethereum", "btc", "eth",
}

# ── Swarm Settings ────────────────────────────────────────────────────────────
SWARM_CANDIDATE_N  = 20   # pass top N markets by volume to the swarm
CONSENSUS_TOP_N    = 5    # final picks to act on
MODEL_TIMEOUT_S    = 60   # seconds before declaring a model call timed out

# Budget split: swarm bot uses this fraction of the total CLOB wallet.
# ★ 2026-05-14: raised 0.50 → 1.00. Crypto loop is now trade-frozen
#   (max_clob_price=0.0 in crypto_params.yaml, fade/settle_arb disabled); it stays
#   alive only to run AR (auto-redeem CTF every 12h). Swarm gets the full CLOB
#   wallet for bet sizing. Revert to 0.50 when re-enabling crypto trading.
SWARM_WALLET_SHARE = 1.00

# ── Loop ──────────────────────────────────────────────────────────────────────
CYCLE_SECONDS = 300   # 5-minute cycle

# ── Gamma API ─────────────────────────────────────────────────────────────────
GAMMA_BASE = "https://gamma-api.polymarket.com"
