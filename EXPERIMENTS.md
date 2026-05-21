# Polymarket Auto-Trading System

A Python bot that takes positions on Polymarket prediction markets, paired with a
funding-rate arbitrage bot on Binance. Real money, real CLOB orders, live since
March 2026.

This document records what we tried, what we kept, and why.

---

## What is running today

| System | Purpose | Capital | Status |
|---|---|---|---|
| Arbitrage (Binance funding rate) | Delta-neutral funding-rate harvest | ~$60 position | LIVE |
| Crypto 5-min up/down | Crowd-momentum scalp on Polymarket BTC/ETH/SOL 5-min markets | Live USDC.e | LIVE |
| AI Swarm | Whale-triggered, LLM-gated bets on long-form prediction markets | Live USDC.e | LIVE |
| Dashboard | Real-time PnL, positions, signal log, model accuracy | localhost:8765 | LIVE |

---

## 1. Arbitrage — funding rate (replaced Hyperliquid breakout)

We initially ran a Hyperliquid-style momentum breakout bot ("HL loop") on SOL +
AVAX perp markets.

**Why we killed HL.** After 995 trades on the post-filter subset:

- Win rate 25.8% vs 25% breakeven — barely above noise.
- At reduced position size ($5 margin / $15 notional), a winning trade paid +$0.018
  and a loser cost $0.006. Even in hot streaks the daily ceiling was ~$1.20.
- Block-to-block variance was huge (6% → 40% → 16% → 40%).
- The crypto 5-min loop was already producing ~$2.33/day at 77% WR — clearly
  dominant for the same capital.

So we replaced HL with a **Binance funding-rate arbitrage** bot:

- Multi-symbol scan (BTC / ETH / SOL / BNB) every 5 minutes.
- Enters delta-neutral (long spot + short perp) when funding rate exceeds
  10.95% APY (0.01% per 8h).
- 4-hour minimum hold to prevent thrashing around the threshold.
- Approach alert at 70% of threshold via ntfy.
- Restart-safe funding collection (counts elapsed periods after downtime).

Live since 2026-04-19. Reverse arb (negative funding) is implemented but gated on
a Binance Margin account.

---

## 2. Crypto 5-min up/down

Polymarket lists "Will BTC be UP or DOWN in the next 5 minutes" markets. Each
window is 300 seconds. We trade these on the **crowd-momentum / flow** thesis:
when one side of the order book is buying aggressively in the last minute of a
window, the resolution will usually agree.

### Signal

Weighted sum of order-flow features, capped to [-1, 1]:

```
score = w_drift   * price_drift          (Binance spot drift vs window open)
      + w_cvd     * cumulative_volume_delta
      + w_clob    * cross_orderbook_imbalance
      + w_oracle  * (binance_price − chainlink_price)  ← settlement-time predictor
      + w_hawkes  * decayed_buy_minus_sell_intensity
      + w_mlofi   * multi_level_order_flow_imbalance
      + ...
```

Direction = `UP if score > 0`, fired when `|score| > threshold` and the entry
window is open (200–270s into the 300s window — that is, the last 30–100s).

### What we experimented with

| Experiment | Result | Decision |
|---|---|---|
| Wider CLOB band [0.68–0.95] | [0.80–0.90) had 76% WR but needed 82% to break even — structurally negative | Tightened to [0.75–0.80) |
| 15-min / 1-hour / 1-day timeframes | 15m: 4 failed strategies (47%–52% WR). 13.5min of price info already priced in at entry. | All killed 2026-05-03 |
| Score cap `\|score\| ≥ 0.65 → skip` | Audit on 7,862 settled trades: low-conviction [0.00, 0.25) zone produced **+$12.90/trade**; extreme [0.65+) produced **+$0.11/trade** despite 88% WR. The cap was backwards. | Cap removed |
| Force-UP at UTC 23 ("Asian open" rule) | Original comment claimed 86% UP rate, n=94. Real audit on 268 windows: 41.4% UP rate — actually the 3rd most DOWN-biased hour. YES bets at UTC 23: –$398.55 PnL on 274 trades. | Rule deleted |
| ML entry filter (LightGBM) | +2.7pp WR on 289 in-band trades, walk-forward validated | Kept as gate at threshold 0.65 |
| Variable bet sizing (Kelly × score) | At $200+ wallet, 0.15× Kelly over-sized vs realized edge | Reverted to flat $5 / trade |
| Fade mode (flip direction on extreme score, last 60s) | Bet small to win 3–9× — explored | Currently disabled (config-flagged) |
| Spot trend filter (block trades against 15m BTC trend) | NEUTRAL trend: 79% WR · UP/DOWN-aligned: 61–65% WR | Block ALL non-NEUTRAL |

### What works

| Zone | n | WR | EV/trade |
|---|---|---|---|
| [0.75–0.80) | 226 | 82.3% | **+$0.227** (dominant profit zone) |
| [0.72–0.75) | 162 | 74.7% | +$0.006 (degrading, removed) |
| [0.80–0.90) | 100+ | 76–87% | negative (structural — lagging features) |

**Strategic principle.** When a price band shows poor EV, the answer is to fix
the signal so it works in that band, not to filter the band away. Filters prevent
losses but equally prevent wins. The current band filter is a placeholder until
leading-indicator features (sub-1m Binance OFI, large-order detection) can
capture information before the crowd absorbs it.

---

## 3. AI Swarm — whale-triggered, LLM-gated

This bot takes positions on **long-form** prediction markets (sports, politics,
events), not crypto price markets.

### Original architecture: 5-agent voting panel

1. **Whale signal** — detect a large CLOB sweep in the last 5 minutes. Only
   markets with whale activity proceed.
2. **RAG context** — fetch DuckDuckGo web search results (news, injury reports,
   polls) for the market question.
3. **Blind vote** — 5 LLMs (deepseek, claude-haiku, claude-sonnet, gpt-4o, qwen)
   each vote `YES / NO / NO_TRADE` from the RAG context alone — whale direction is
   intentionally hidden so models can independently disagree with the whale.
4. **Alignment** — model majority must match whale direction.
5. **Synthesis (Design A)** — deepseek sees the anonymised panel reasoning + RAG
   and produces a final calibrated verdict.
6. **Execute** if all quality gates pass: score ≥ 0.60, avg_conf ≥ 70,
   ai_agree ≥ 0.50, NO-direction only, ask in [0.65, 0.95].

### Why we moved toward "deepseek-solo" effectively

Per-market cost was ~6 LLM calls (5 panel + 1 synthesis). For ~20 candidate
markets per cycle and a 5-minute cycle, that compounded fast. We tracked which
models actually added information:

| Model | Accuracy when voting | Abstain rate | Notes |
|---|---|---|---|
| **deepseek** | **67.2%** (45W/22L) | **64.4%** | Abstains 87% of losses — the conservative oracle |
| claude-sonnet | 58.1% (50W/36L) | 55.0% | Selective but lower accuracy when voting |
| gpt-4o | 56.8% (88W/67L) | 17.6% | High participation, lowest accuracy |
| claude-haiku | — | — | Cheap participant (replaced claude-opus 2026-04-12: 18× cost for same 3-line task) |
| qwen | — | — | Diversity, accuracy not yet measured |

Deepseek alone has the highest WR and the highest abstain-on-losses rate — it is
the only model that knows when to stay home. When deepseek's API credits
exhausted on 2026-05-13, we disabled the synthesis (Design A) round and fell back
to **deepseek round-1 veto** as the operational quality gate. The 4 other models
continue to vote (for alignment and conf), but deepseek is the gate.

This is the de-facto "deepseek-solo" mode: cheap participation models gather
signal, deepseek decides whether to take the trade.

### Static vs dynamic bet sizing

**Dynamic (tried 2026-04-12):**
- `bet = base × (0.40 × score_norm + 0.60 × synthesis_conf_norm)`
- Higher conviction → bigger bet, up to 35% of swarm budget.
- Idea: score sweet-spot [0.75–0.80) had 94% WR and EV = 0.183, the same as
  score=0.65 (77% WR, EV = 0.05). Flat sizing left money on the table.

**Why we reverted to static $20 flat:**
- After 120 settled real trades, the WR-vs-score curve was not monotonic:
  score [0.30–0.40) had 84% WR vs score [0.40–0.50) at 72%. Higher score did not
  mean higher WR.
- Dynamic sizing was rewarding the wrong tail.
- Operational simplicity: `bet = max(10, min(20, balance × 0.35))` —
  no surprises, easy to audit, no overfitting to small-sample buckets.

We may revisit dynamic sizing once we have a calibrated `P(win)` model rather
than raw consensus score.

### What works (NO direction only)

- All-time YES: 39.0% WR, –$109 PnL over 82 trades → **YES blocked entirely**.
- All-time NO: 81.9% WR, +$19.27 PnL over 105 trades.
- The bot now only takes NO positions in [0.65–0.95] ask range.

---

## 4. Dashboard

Local web dashboard on port 8765 (also bound to 0.0.0.0 for remote access),
threaded `HTTPServer` so a stuck connection cannot block other clients.

Surfaces:

- Live wallet balance (CLOB + pending CTF tokens + Polygon POL for gas)
- Real-mode PnL across crypto and swarm, broken down by direction, score, hour
- Recent fills + open positions
- Per-model accuracy leaderboard (which LLM is voting correctly this week)
- Signal log and ghost-band data (hypothetical trades in excluded zones, for
  future EV analysis)
- Funding-arb status (active symbol, hours-to-collect, total funding collected)

---

## Experimented but not running

### Conviction bot
A second crypto strategy that bets in inverse proportion to score — high
conviction = small bet, low conviction = full bet. Currently in paper mode,
accumulating data. Motivation: the 7,862-trade audit (see the score-cap
experiment above) found low-conviction is structurally most profitable.

### 15-min / 1-hour / 1-day crypto loops
All killed 2026-05-03 after 4 failed 15m strategies. Root cause: on a 15-min
window, the market has 13.5 min of price information already priced into the
fill by the time we enter — there is no residual edge. The 5-min window works
because only 3.5 min is priced in.

### Strike scanner
A Black-Scholes "edge" detector for Polymarket strike-price options. 18 settled
tickets, 2 wins (11% WR), –$14.66 net. The 6–21% theoretical edge was not
predictive of outcomes. Killed 2026-05-11.

### Settlement arbitrage
A latency-arb idea: at T-30s before window close, compare CLOB price for the
predicted-winning token to the Chainlink-vs-window-open math; take the side the
math implies if CLOB underprices it. Implemented, run briefly, killed 2026-05-14
during wallet reallocation to give swarm full budget.

### Main bot (general Polymarket signal scanner)
The original architecture: scan all categories (politics, sports, crypto,
other), ensemble logistic + tree + LLM model. Permanently stopped — sports went
to 0% WR, politics to 17%. The training data was contaminated (features built
at settlement time leaked the outcome), so the ML weights are still pinned at
LLM-90% / ML-5%. Code preserved for reference.

---

## Repository layout

```
poly-model/
├── src/
│   ├── crypto/              ← 5-min flow loop (loop.py, flow.py, clob_feed.py, redeem.py)
│   ├── crypto_main.py       ← Entry point — screen -S crypto
│   ├── crypto_conviction_main.py  ← Conviction bot (paper)
│   ├── dashboard.py         ← Web dashboard
│   └── ...
├── bot/                     ← AI Swarm
│   ├── main.py              ← Entry point — screen -S swarm
│   ├── swarm/
│   │   ├── consensus.py     ← Score formula + quality gates
│   │   ├── models.py        ← Blind-vote prompt + synthesis judge
│   │   └── search.py        ← DuckDuckGo RAG (subprocess-isolated for primp segfault)
│   ├── execution.py         ← CLOB orders + direction/ask filters
│   └── paper.py             ← Real-trade tracker + model accuracy leaderboard
├── funding_arb/
│   ├── main.py              ← Entry point — screen -S arb
│   └── src/                 ← futures_client, spot_client, risk, position_manager
├── config/                  ← All tunable parameters (yaml)
├── data/                    ← State files + signal logs + ghost bands
├── scripts/                 ← Backtests, retraining, on-chain redemption
└── logs/                    ← Per-strategy log files
```

---

## Strategic principles (lessons learned)

1. **Fix the signal, not the band.** Filtering a zone with poor EV prevents the
   upside in that zone too. Treat band filters as placeholders, not endpoints.
2. **WR alone is misleading.** At fill 0.90 you need 92% WR to break even. The
   right metric is direction-aware EV per trade on executed fills — not
   signal-log skipped-signal WR (which is inflated).
3. **Cheap models gather signal; expensive models gate.** Don't pay GPT-4o-class
   prices to vote when deepseek is twice as good at abstaining on losers.
4. **Static beats dynamic until you have a calibrated probability.** Variable
   bet sizing rewards wrong-tail noise on small samples. Use flat sizing until
   `P(win)` is calibrated, not just "score is higher."
5. **Documented experiments survive their authors.** Every change has a paired
   entry in the project notes with the reason and the data — when you're
   tempted to undo a filter, you can read why it was added.

---

## Operations

The bots run 24/7 on an Oracle Cloud VM under `screen` sessions:
`crypto`, `swarm`, `arb`, `dashboard`, `conviction`. Restart commands,
on-chain redemption flow, and Polygon wallet setup are in
[`docs/server_setup.md`](docs/server_setup.md).

---

## Disclaimer

This is a personal research project. Prediction-market trading is
high-variance and the strategies here have lost real money during their
development as well as made it. Read the killed-system post-mortems above
before reusing any of this code.
