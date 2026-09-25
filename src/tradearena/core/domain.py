from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class RiskPhase(str, Enum):
    PRE_TRADE = "pre_trade"
    IN_TRADE = "in_trade"
    POST_TRADE = "post_trade"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class NewsItem:
    timestamp: datetime
    source: str
    title: str
    body: str
    sentiment: float = 0.0
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class MacroPoint:
    timestamp: datetime
    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True)
class FilingItem:
    timestamp: datetime
    source: str
    form_type: str
    title: str
    body: str
    sentiment: float = 0.0
    symbols: tuple[str, ...] = ()
    accession: str = ""


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: datetime
    bars: dict[str, Bar]
    news: tuple[NewsItem, ...] = ()
    macro: tuple[MacroPoint, ...] = ()
    filings: tuple[FilingItem, ...] = ()
    alt_data: dict[str, Any] = field(default_factory=dict)

    def price(self, symbol: str) -> float:
        return self.bars[symbol].close


@dataclass(frozen=True)
class Signal:
    symbol: str
    score: float
    horizon: str
    confidence: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    symbol: str
    side: Side
    target_weight: float
    confidence: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    requested_quantity: float | None = None
    latency_steps: int = 0
    liquidity_available: float | None = None
    fill_ratio: float = 1.0
    slippage: float = 0.0
    status: str = "filled"

    @property
    def signed_quantity(self) -> float:
        if self.side == Side.BUY:
            return self.quantity
        if self.side == Side.SELL:
            return -self.quantity
        return 0.0


@dataclass(frozen=True)
class RiskCheck:
    name: str
    passed: bool
    severity: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskBudget:
    max_position_weight: float = 0.35
    max_gross_exposure: float = 1.0
    max_single_step_turnover: float = 0.75
    max_drawdown: float = 0.2
    max_order_participation: float = 0.05
    min_confidence: float = 0.05
    max_latency_steps: int = 2
    max_slippage_bps: float = 50.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskViolation:
    phase: RiskPhase
    constraint: str
    severity: str
    observed: float | str
    limit: float | str
    message: str
    symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAttribution:
    timestamp: datetime
    realized_pnl: float
    commission: float
    slippage_cost: float
    exposure: dict[str, float]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskReport:
    timestamp: datetime
    checks: tuple[RiskCheck, ...]
    approved_count: int
    blocked_count: int
    clipped_count: int
    phase: RiskPhase = RiskPhase.PRE_TRADE
    budget: RiskBudget | None = None
    violations: tuple[RiskViolation, ...] = ()
    attribution: RiskAttribution | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.severity == "error")


@dataclass(frozen=True)
class ExecutionReport:
    timestamp: datetime
    submitted_orders: int
    eligible_orders: int
    filled_orders: int
    partial_fills: int
    pending_orders: int
    rejected_orders: int
    total_commission: float
    total_slippage: float
    average_latency_steps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    timestamp: datetime
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReproducibilityState:
    prompt_version: str
    model_version: str
    retrieved_documents: tuple[str, ...]
    market_data_timestamp: datetime
    tool_outputs: tuple[ToolCallRecord, ...]
    memory_digest: str
    risk_constraints: RiskBudget
    portfolio_state: dict[str, Any]
    agent_discussion_history: tuple[str, ...] = ()
    execution_simulator_state: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 0


@dataclass(frozen=True)
class AgentProtocolTrace:
    observation_schema: dict[str, Any]
    memory_schema: dict[str, Any]
    tool_schema: dict[str, Any]
    action_schema: dict[str, Any]
    risk_schema: dict[str, Any]
    trajectory_schema: dict[str, Any]
    evaluation_schema: dict[str, Any]
    observe: dict[str, Any]
    plan: dict[str, Any]
    propose_order: dict[str, Any]
    risk_report: dict[str, Any]
    revise: dict[str, Any]
    act: dict[str, Any]
    reflect: dict[str, Any]


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def market_value(self) -> float:
        return sum(qty * self.last_prices.get(symbol, 0.0) for symbol, qty in self.positions.items())

    def equity(self) -> float:
        return self.cash + self.market_value()

    def weight(self, symbol: str) -> float:
        equity = self.equity()
        if equity == 0:
            return 0.0
        return self.positions.get(symbol, 0.0) * self.last_prices.get(symbol, 0.0) / equity

    def copy(self) -> PortfolioState:
        return PortfolioState(
            cash=self.cash,
            positions=dict(self.positions),
            last_prices=dict(self.last_prices),
            realized_pnl=self.realized_pnl,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    initial_cash: float = 100_000.0
    symbols: tuple[str, ...] = ("SYN",)
    seed: int = 7
    # Optional warm-start holdings as target weights per symbol. When set, the
    # portfolio is seeded with these positions at the first bar's prices before
    # the agent acts, so initial-construction execution cost is not charged to
    # the agent (used for buy-and-hold anchor robustness checks).
    initial_holdings_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
