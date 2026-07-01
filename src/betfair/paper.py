"""
src/betfair/paper.py — Paper trading state for the Betfair football bot.

Tracks virtual BACK bets without placing real orders. Settles from Betfair API
results (winning selection_id). PnL formula:
  BACK win:  stake * (odds - 1) * (1 - commission)
  BACK loss: -stake

Each bet logs the fields needed to replicate the Polymarket edge decomposition later
(implied prob, vote breakdown, vote margin, liquidity, minutes to kickoff).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PaperBet:
    bet_id:       str
    market_id:    str
    match_name:   str       # "France v Sweden"
    competition:  str
    runner_id:    str       # winning-selection id we backed
    outcome:      str       # "HOME" / "DRAW" / "AWAY"
    runner_name:  str       # team name or "The Draw"
    side:         str       # "BACK"
    odds:         float
    stake:        float     # GBP
    commission:   float
    placed_ts:    float
    # signal metadata (for edge analysis)
    implied_prob: float = 0.0   # 1/odds of the backed outcome — break-even WR
    agree_frac:   float = 0.0
    avg_conf:     float = 0.0
    swarm_score:  float = 0.0
    vote_home:    int   = 0
    vote_draw:    int   = 0
    vote_away:    int   = 0
    vote_abstain: int   = 0
    vote_margin:  int   = 0     # plurality - runner-up (price-orthogonal edge proxy)
    total_matched: float = 0.0  # market liquidity (GBP) at entry
    mins_to_ko:   float = 0.0
    # set on settlement:
    settled:      bool  = False
    won:          bool  = False
    pnl:          float = 0.0
    settled_ts:   float = 0.0
    winner_name:  str   = ""


@dataclass
class BetfairPaperState:
    open_bets:   list[dict] = field(default_factory=list)
    closed_bets: list[dict] = field(default_factory=list)
    total_pnl:   float = 0.0
    trade_count: int   = 0
    win_count:   int   = 0


def _load_state(path: str) -> BetfairPaperState:
    if not os.path.exists(path):
        return BetfairPaperState()
    try:
        with open(path) as f:
            d = json.load(f)
        s = BetfairPaperState()
        s.open_bets   = d.get("open_bets", [])
        s.closed_bets = d.get("closed_bets", [])
        s.total_pnl   = d.get("total_pnl", 0.0)
        s.trade_count = d.get("trade_count", 0)
        s.win_count   = d.get("win_count", 0)
        return s
    except Exception as e:
        log.warning("Failed to load betfair state from %s: %s", path, e)
        return BetfairPaperState()


def _save_state(state: BetfairPaperState, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, path)


class PaperTrader:
    def __init__(self, state_path: str, paper_log_path: str, commission: float = 0.05) -> None:
        self._state_path = state_path
        self._paper_log  = paper_log_path
        self._commission = commission
        self._state      = _load_state(state_path)
        self._bet_counter = self._state.trade_count + 1
        log.info(
            "PaperTrader: %d open, %d closed, PnL=GBP %.2f",
            len(self._state.open_bets), len(self._state.closed_bets), self._state.total_pnl,
        )

    def place(
        self,
        market_id: str,
        match_name: str,
        competition: str,
        runner_id: str,
        outcome: str,
        runner_name: str,
        odds: float,
        stake: float,
        implied_prob: float = 0.0,
        agree_frac: float = 0.0,
        avg_conf: float = 0.0,
        swarm_score: float = 0.0,
        vote_home: int = 0,
        vote_draw: int = 0,
        vote_away: int = 0,
        vote_abstain: int = 0,
        vote_margin: int = 0,
        total_matched: float = 0.0,
        mins_to_ko: float = 0.0,
    ) -> str:
        bet_id = f"BF{int(time.time())}-{self._bet_counter}"
        self._bet_counter += 1
        bet = PaperBet(
            bet_id=bet_id, market_id=market_id, match_name=match_name,
            competition=competition, runner_id=runner_id, outcome=outcome,
            runner_name=runner_name, side="BACK", odds=odds, stake=stake,
            commission=self._commission, placed_ts=time.time(),
            implied_prob=implied_prob, agree_frac=agree_frac, avg_conf=avg_conf,
            swarm_score=swarm_score, vote_home=vote_home, vote_draw=vote_draw,
            vote_away=vote_away, vote_abstain=vote_abstain, vote_margin=vote_margin,
            total_matched=total_matched, mins_to_ko=mins_to_ko,
        )
        self._state.open_bets.append(asdict(bet))
        _save_state(self._state, self._state_path)
        log.info(
            "PAPER BET %s: BACK %s (%s) @ %.2f GBP%.2f  [%s | agree=%.0f%% conf=%.0f margin=%d liq=GBP%.0fk]",
            bet_id, runner_name, outcome, odds, stake, match_name,
            agree_frac * 100, avg_conf, vote_margin, total_matched / 1000.0,
        )
        return bet_id

    def settle_market(self, market_id: str, winner_runner_id: str, winner_name: str) -> None:
        """Settle all open bets for a market given the winning selection."""
        still_open: list[dict] = []
        for bet_d in self._state.open_bets:
            if bet_d["market_id"] != market_id:
                still_open.append(bet_d)
                continue

            is_winner = (bet_d["runner_id"] == winner_runner_id)
            odds  = bet_d["odds"]
            stake = bet_d["stake"]
            comm  = bet_d["commission"]

            if is_winner:
                pnl = stake * (odds - 1) * (1.0 - comm)
                won = True
            else:
                pnl = -stake
                won = False

            bet_d.update({
                "settled": True, "won": won, "pnl": round(pnl, 4),
                "settled_ts": time.time(), "winner_name": winner_name,
            })
            self._state.closed_bets.append(bet_d)
            self._state.total_pnl += pnl
            self._state.trade_count += 1
            if won:
                self._state.win_count += 1

            result_str = f"+GBP{pnl:.2f}" if pnl >= 0 else f"-GBP{abs(pnl):.2f}"
            log.info(
                "SETTLE %s: BACK %s @ %.2f -> %s %s  [total PnL=GBP%.2f  WR=%.0f%%]",
                bet_d["bet_id"], bet_d["runner_name"], odds,
                "WIN" if won else "LOSS", result_str,
                self._state.total_pnl,
                100 * self._state.win_count / max(self._state.trade_count, 1),
            )
            self._append_log(bet_d)

        self._state.open_bets = still_open
        _save_state(self._state, self._state_path)

    def _append_log(self, bet_d: dict) -> None:
        os.makedirs(os.path.dirname(self._paper_log) if os.path.dirname(self._paper_log) else ".", exist_ok=True)
        with open(self._paper_log, "a") as f:
            f.write(json.dumps(bet_d) + "\n")

    @property
    def stats(self) -> dict:
        trades = self._state.trade_count
        wr = self._state.win_count / trades if trades else 0.0
        return {
            "open":      len(self._state.open_bets),
            "closed":    trades,
            "win_count": self._state.win_count,
            "wr":        wr,
            "total_pnl": round(self._state.total_pnl, 4),
        }

    def already_bet_market(self, market_id: str) -> bool:
        return any(b["market_id"] == market_id for b in self._state.open_bets + self._state.closed_bets)

    def open_bet_market_ids(self) -> list[str]:
        return [b["market_id"] for b in self._state.open_bets]
