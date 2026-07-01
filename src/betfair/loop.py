"""
src/betfair/loop.py — Betfair FOOTBALL (soccer) MATCH_ODDS paper trading loop.

Pivoted from horse racing 2026-06-30 (see config/betfair_params.yaml header).

Per cycle:
  - Scan soccer MATCH_ODDS markets kicking off in [min,max] minutes, kept only if
    liquidity (total_matched) >= min_liquidity_gbp  → auto-selects World Cup / top leagues.
  - For each match: map the 3 selections to HOME/DRAW/AWAY, fetch best-back odds,
    pull RAG team-news context, run the LLM swarm blind, back the plurality outcome
    if the agree/confidence/score/odds gates pass.
  - Settle when Betfair marks the market CLOSED with a WINNER.

Paper mode only (virtual_mode=true): virtual bets in data/betfair_paper.jsonl.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import Optional

import yaml

from src.betfair.paper import PaperTrader
from src.betfair.signal import MatchOutcome, ask_match_swarm, compute_consensus

log = logging.getLogger("betfair.loop")


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_cfg(path: str = "config/betfair_params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Betfair API helpers ────────────────────────────────────────────────────────

def _login(username: str, password: str, app_key: str):
    import betfairlightweight
    client = betfairlightweight.APIClient(username=username, password=password, app_key=app_key)
    client.login_interactive()
    return client


def _is_session_error(exc: BaseException) -> bool:
    s = str(exc)
    return ("INVALID_SESSION_INFORMATION" in s or "ANGX-0003" in s or "NO_SESSION" in s)


def _ensure_session(holder: dict) -> None:
    log.warning("Betfair session expired — re-logging in")
    holder["trading"] = _login(*holder["creds"])
    holder["last_keepalive"] = time.time()
    log.info("Betfair re-login OK")


def _keep_alive_if_due(holder: dict, interval_s: int = 1800) -> None:
    if time.time() - holder.get("last_keepalive", 0) < interval_s:
        return
    try:
        holder["trading"].keep_alive()
        holder["last_keepalive"] = time.time()
    except Exception as e:
        log.warning("keep_alive failed: %s", e)
        try:
            _ensure_session(holder)
        except Exception as e2:
            log.error("re-login failed during keep_alive: %s", e2)


def _upcoming_markets(holder: dict, cfg: dict) -> list:
    """Soccer MATCH_ODDS markets kicking off soon AND liquid enough to be news-covered."""
    from betfairlightweight.filters import market_filter, time_range

    now = datetime.datetime.utcnow()
    lo  = now + datetime.timedelta(minutes=cfg["min_minutes_before_ko"])
    hi  = now + datetime.timedelta(minutes=cfg["max_minutes_before_ko"])

    mf_kwargs = dict(
        event_type_ids=cfg["event_type_ids"],
        market_type_codes=cfg["market_type_codes"],
        market_start_time=time_range(from_=lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     to=hi.strftime("%Y-%m-%dT%H:%M:%SZ")),
    )
    if cfg.get("market_countries"):
        mf_kwargs["market_countries"] = cfg["market_countries"]
    flt = market_filter(**mf_kwargs)

    def _do():
        return holder["trading"].betting.list_market_catalogue(
            filter=flt,
            market_projection=["EVENT", "COMPETITION", "MARKET_START_TIME", "RUNNER_DESCRIPTION"],
            sort="FIRST_TO_START",
            max_results=50,
        )

    try:
        cat = _do() or []
    except Exception as e:
        if _is_session_error(e):
            try:
                _ensure_session(holder)
                cat = _do() or []
            except Exception as e2:
                log.warning("list_market_catalogue retry failed: %s", e2)
                return []
        else:
            log.warning("list_market_catalogue error: %s", e)
            return []

    min_liq = cfg.get("min_liquidity_gbp", 0)
    return [m for m in cat if (m.total_matched or 0) >= min_liq]


def _best_back_by_selection(holder: dict, market_id: str) -> dict:
    """Return {selection_id(str): best_back_price} for ACTIVE runners."""
    def _do():
        return holder["trading"].betting.list_market_book(
            market_ids=[market_id],
            price_projection={"priceData": ["EX_BEST_OFFERS"]},
        )

    try:
        books = _do()
    except Exception as e:
        if _is_session_error(e):
            try:
                _ensure_session(holder)
                books = _do()
            except Exception as e2:
                log.warning("list_market_book retry failed for %s: %s", market_id, e2)
                return {}
        else:
            log.warning("list_market_book error for %s: %s", market_id, e)
            return {}
    if not books:
        return {}
    out = {}
    for r in books[0].runners:
        if r.status != "ACTIVE":
            continue
        price = None
        if r.ex and r.ex.available_to_back:
            price = r.ex.available_to_back[0].price
        out[str(r.selection_id)] = price
    return out


def _get_winner(holder: dict, market_id: str) -> Optional[tuple[str, str]]:
    """Return (winner_selection_id, name) if the market is CLOSED with a winner."""
    def _do():
        return holder["trading"].betting.list_market_book(
            market_ids=[market_id], price_projection={"priceData": []},
        )

    try:
        books = _do()
    except Exception as e:
        if _is_session_error(e):
            try:
                _ensure_session(holder)
                books = _do()
            except Exception:
                return None
        else:
            return None
    if not books or books[0].status != "CLOSED":
        return None
    for r in books[0].runners:
        if r.status == "WINNER":
            return str(r.selection_id), getattr(r, "runner_name", "")
    return None


# ── Outcome mapping ────────────────────────────────────────────────────────────

def _resolve_outcomes(catalogue) -> Optional[dict]:
    """
    Map the 3 MATCH_ODDS selections to HOME / DRAW / AWAY.
    Returns {"HOME": (sid,name), "DRAW": (sid,name), "AWAY": (sid,name)} or None.
    """
    event_name = (catalogue.event.name or "") if catalogue.event else ""
    if " v " not in event_name:
        return None
    home_name, away_name = (p.strip() for p in event_name.split(" v ", 1))

    runners = []
    for r in catalogue.runners:
        sp = getattr(r, "sort_priority", 99) or 99
        runners.append((str(r.selection_id), (r.runner_name or "").strip(), sp))

    draw = next((x for x in runners if x[1].lower() == "the draw"), None)
    others = [x for x in runners if x is not draw]
    if draw is None or len(others) != 2:
        return None

    def _matches(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return bool(a) and bool(b) and (a in b or b in a)

    home = away = None
    for x in others:
        if _matches(x[1], home_name):
            home = x
        elif _matches(x[1], away_name):
            away = x
    # fallback: assign by sort_priority (lower = home)
    if home is None or away is None:
        others.sort(key=lambda x: x[2])
        home, away = others[0], others[1]

    return {
        "HOME": (home[0], home[1]),
        "DRAW": (draw[0], draw[1]),
        "AWAY": (away[0], away[1]),
    }


# ── RAG search ────────────────────────────────────────────────────────────────

def _fetch_rag(home: str, away: str, competition: str) -> str:
    from bot.swarm.search import fetch_market_context
    query = (f"{home} vs {away} {competition} predicted lineup team news injuries "
             f"preview {datetime.date.today().isoformat()}")
    return fetch_market_context(query, hours_to_close=1.0)


# ── Per-match evaluation ───────────────────────────────────────────────────────

async def _evaluate_market(holder: dict, catalogue, cfg: dict, paper: PaperTrader) -> None:
    market_id = catalogue.market_id
    if paper.already_bet_market(market_id):
        return

    mapping = _resolve_outcomes(catalogue)
    if mapping is None:
        return

    event_name = catalogue.event.name or ""
    competition = catalogue.competition.name if catalogue.competition else ""
    home_name = mapping["HOME"][1]
    away_name = mapping["AWAY"][1]

    prices = _best_back_by_selection(holder, market_id)
    odds = {}
    for key in ("HOME", "DRAW", "AWAY"):
        sid = mapping[key][0]
        p = prices.get(sid)
        if not p or p < 1.01:
            return  # missing/suspended price — skip whole match
        odds[key] = p

    now = datetime.datetime.now(datetime.timezone.utc)
    mst = catalogue.market_start_time
    if mst is not None:
        if mst.tzinfo is None:                       # normalize: Betfair usually returns tz-aware UTC
            mst = mst.replace(tzinfo=datetime.timezone.utc)
        mins_to_ko = (mst - now).total_seconds() / 60.0
    else:
        mins_to_ko = 0.0
    liquidity = catalogue.total_matched or 0.0

    log.info("Evaluating %s [%s] KO in %.0fm  liq=GBP%.0fk  (H %.2f / D %.2f / A %.2f)",
             event_name, competition, mins_to_ko, liquidity / 1000.0,
             odds["HOME"], odds["DRAW"], odds["AWAY"])

    rag_ctx = _fetch_rag(home_name, away_name, competition)

    verdicts = await ask_match_swarm(
        home=home_name, away=away_name, competition=competition,
        odds_home=odds["HOME"], odds_draw=odds["DRAW"], odds_away=odds["AWAY"],
        mins_to_ko=mins_to_ko, rag_context=rag_ctx,
    )
    con = compute_consensus(verdicts,
                            min_agree_frac=cfg["min_ai_agree_frac"],
                            min_confidence=cfg["min_confidence"])

    if con.outcome is None:
        log.info("  -> NO_TRADE (votes=%s agree=%.0f%% conf=%.0f margin=%d)",
                 con.votes, con.agree_frac * 100, con.avg_conf, con.margin)
        return

    swarm_score = con.agree_frac * (con.avg_conf / 100.0)
    if swarm_score < cfg["min_swarm_score"]:
        log.info("  -> %s score=%.2f below %.2f — skip",
                 con.outcome.value, swarm_score, cfg["min_swarm_score"])
        return

    backed_odds = odds[con.outcome.value]
    if backed_odds < cfg["min_odds_back"] or backed_odds > cfg["max_odds_back"]:
        log.info("  -> %s @ %.2f outside odds range [%.2f,%.2f] — skip",
                 con.outcome.value, backed_odds, cfg["min_odds_back"], cfg["max_odds_back"])
        return

    sid, runner_name = mapping[con.outcome.value]
    paper.place(
        market_id=market_id, match_name=event_name, competition=competition,
        runner_id=sid, outcome=con.outcome.value, runner_name=runner_name,
        odds=backed_odds, stake=cfg["bet_size_gbp"],
        implied_prob=1.0 / backed_odds,
        agree_frac=con.agree_frac, avg_conf=con.avg_conf, swarm_score=swarm_score,
        vote_home=con.votes["HOME"], vote_draw=con.votes["DRAW"],
        vote_away=con.votes["AWAY"], vote_abstain=con.votes["NO_TRADE"],
        vote_margin=con.margin, total_matched=liquidity, mins_to_ko=mins_to_ko,
    )


# ── Settlement ──────────────────────────────────────────────────────────────────

def _settle_pending(holder: dict, paper: PaperTrader) -> None:
    for mid in list(set(paper.open_bet_market_ids())):
        result = _get_winner(holder, mid)
        if result:
            winner_id, winner_name = result
            log.info("Market %s CLOSED — winner: %s (%s)", mid, winner_name, winner_id)
            paper.settle_market(mid, winner_id, winner_name)


# ── Main run loop ────────────────────────────────────────────────────────────────

def run(username: str, password: str, app_key: str,
        virtual_mode: bool = True, cfg_path: str = "config/betfair_params.yaml") -> None:
    cfg = _load_cfg(cfg_path)
    virtual_mode = virtual_mode or cfg.get("virtual_mode", True)
    mode_str = "VIRTUAL (paper)" if virtual_mode else "LIVE (real bets)"

    log.info("=" * 60)
    log.info("Betfair FOOTBALL bot — mode=%s", mode_str)
    log.info("Soccer MATCH_ODDS | liquidity >= GBP%.0fk | KO window %.0f-%.0f min",
             cfg["min_liquidity_gbp"] / 1000.0,
             cfg["min_minutes_before_ko"], cfg["max_minutes_before_ko"])
    log.info("Gates: agree>=%.0f%% conf>=%.0f score>=%.2f | back odds [%.2f,%.2f]",
             cfg["min_ai_agree_frac"] * 100, cfg["min_confidence"],
             cfg["min_swarm_score"], cfg["min_odds_back"], cfg["max_odds_back"])
    log.info("=" * 60)

    paper = PaperTrader(state_path=cfg["state_file"], paper_log_path=cfg["paper_log"],
                        commission=cfg["commission_rate"])

    holder: dict = {
        "creds": (username, password, app_key),
        "trading": _login(username, password, app_key),
        "last_keepalive": time.time(),
    }
    log.info("Betfair login OK")

    loop = asyncio.new_event_loop()
    last_scan = 0.0

    while True:
        try:
            now = time.time()
            _keep_alive_if_due(holder)
            _settle_pending(holder, paper)

            if now - last_scan >= cfg["market_cache_ttl"]:
                markets = _upcoming_markets(holder, cfg)
                last_scan = now
                st = paper.stats
                log.info("Market scan: %d qualifying matches  |  open=%d  PnL=GBP%.2f  WR=%.0f%%",
                         len(markets), st["open"], st["total_pnl"], st["wr"] * 100)
                for catalogue in markets:
                    if not paper.already_bet_market(catalogue.market_id):
                        try:
                            loop.run_until_complete(_evaluate_market(holder, catalogue, cfg, paper))
                        except Exception as e:
                            log.error("evaluate_market error: %s", e, exc_info=True)

            time.sleep(cfg["poll_interval"])

        except KeyboardInterrupt:
            log.info("Interrupted — shutting down Betfair bot")
            break
        except Exception as e:
            if _is_session_error(e):
                try:
                    _ensure_session(holder)
                except Exception as e2:
                    log.error("Loop-level re-login failed: %s", e2)
                    time.sleep(60)
            else:
                log.error("Loop error: %s", e, exc_info=True)
                time.sleep(60)
