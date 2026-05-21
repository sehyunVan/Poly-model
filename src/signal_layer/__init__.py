"""
signal_layer package public interface.

Named signal_layer (not signal) to avoid shadowing Python's stdlib
signal module, which sklearn/joblib depend on internally.

    from signal_layer import compute_alpha, generate_signals, TradeSignal
    from signal_layer import passes_pre_trade_filters
"""

from .schemas import TradeSignal
from .alpha import compute_alpha, generate_signals
from .filters import passes_pre_trade_filters

__all__ = [
    "TradeSignal",
    "compute_alpha",
    "generate_signals",
    "passes_pre_trade_filters",
]
