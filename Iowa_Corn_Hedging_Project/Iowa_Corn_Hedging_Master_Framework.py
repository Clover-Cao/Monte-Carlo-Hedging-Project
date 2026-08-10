# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Iowa Corn Hedging: Master Python Framework
#
# This file consolidates Lessons 1–11 into one reproducible, English-language
# framework. It is both a normal Python program and a Jupytext-style notebook:
# the `# %%` markers become notebook cells in VS Code, JupyterLab, or the
# companion `.ipynb` file.
#
# The framework answers one decision question:
#
# > Under the stated historical data, calibration, cost, basis, and liquidity
# > assumptions, which December corn futures hedge provides the strongest
# > protection in the lower 5% of farm-profit outcomes without materially
# > sacrificing expected profit or creating excessive over-hedging?
#
# The framework preserves the lessons' timing discipline:
#
# 1. A March hedge can use only the trend yield and the March futures price.
# 2. A July adjustment can use July PDSI, the July yield forecast, and the July
#    futures price, but not the final yield residual.
# 3. Harvest profit uses realized yield, harvest futures, and local basis.
# 4. Every strategy faces the same 10,000 scenarios (common random numbers).

# %%
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# %% [markdown]
# ## 0. Model governance and pre-specified decision framework
#
# All economic and implementation assumptions are centralized below. These
# values match the eleven lessons and the final Excel workbook. They are model
# assumptions—not broker quotes, guarantees, or trading advice.
#
# The preferred strategy is selected only after applying two safeguards:
#
# - **Expected-profit safeguard:** mean profit may not be more than
#   `5% × |unhedged mean profit|` below the unhedged mean.
# - **Over-hedging safeguard:** the probability that hedged bushels exceed
#   actual production may not exceed 10%.
#
# Among eligible strategies, the highest lower-tail 5% profit CVaR wins. Here,
# CVaR is the average profit at or below the 5th percentile, so a larger (less
# negative) value is better. If CVaR ties, higher expected profit breaks the tie.

# %%
@dataclass(frozen=True)
class ModelConfig:
    """Central, auditable assumptions for the complete Lessons 1–11 model."""

    calibration_start_year: int = 1996
    calibration_end_year: int = 2025
    forecast_year: int = 2026
    n_simulations: int = 10_000
    random_seed: int = 8_122_026

    farm_acres: int = 1_000
    production_cost_usd_per_acre: float = 911.98
    preseason_futures_usd_per_bushel: float = 4.70
    futures_price_floor_usd_per_bushel: float = 1.50
    cash_price_floor_usd_per_bushel: float = 0.50

    basis_low_usd_per_bushel: float = -0.28
    basis_mode_usd_per_bushel: float = -0.20
    basis_high_usd_per_bushel: float = -0.11

    contract_size_bushels: int = 5_000
    transaction_cost_usd_per_contract_side: float = 25.00
    initial_margin_usd_per_contract: float = 2_500.00
    annual_margin_financing_rate: float = 0.06
    days_preseason_to_july: int = 136
    days_july_to_harvest: int = 108
    days_per_year: int = 365
    margin_liquidity_reserve_usd: float = 50_000.00

    fixed_hedge_ratios: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)
    adaptive_initial_ratio: float = 0.50
    drought_target_ratio: float = 0.25
    normal_target_ratio: float = 0.50
    wet_target_ratio: float = 0.75

    lower_tail_probability: float = 0.05
    max_mean_profit_shortfall_fraction: float = 0.05
    max_overhedge_probability: float = 0.10
    z_95: float = 1.96

    def __post_init__(self) -> None:
        if self.calibration_end_year < self.calibration_start_year:
            raise ValueError("Calibration end year must not precede the start year.")
        if self.forecast_year <= self.calibration_end_year:
            raise ValueError("Forecast year must follow the calibration period.")
        if self.n_simulations <= 0 or self.farm_acres <= 0:
            raise ValueError("Simulation count and farm acres must be positive.")
        if self.contract_size_bushels <= 0:
            raise ValueError("Contract size must be positive.")
        if not (
            self.basis_low_usd_per_bushel
            <= self.basis_mode_usd_per_bushel
            <= self.basis_high_usd_per_bushel
        ):
            raise ValueError("Triangular basis parameters must satisfy low <= mode <= high.")
        if any(ratio < 0 or ratio > 1 for ratio in self.fixed_hedge_ratios):
            raise ValueError("Hedge ratios must be between zero and one.")
        if not 0 < self.lower_tail_probability < 0.5:
            raise ValueError("Lower-tail probability must be between zero and 0.5.")

    @property
    def forecast_trend_index(self) -> int:
        return self.forecast_year - self.calibration_start_year

    @property
    def total_production_cost_usd(self) -> float:
        return self.farm_acres * self.production_cost_usd_per_acre


ALL_STRATEGIES = (
    "Fixed 0%",
    "Fixed 25%",
    "Fixed 50%",
    "Fixed 75%",
    "Fixed 100%",
    "Updated 50%",
    "Weather-signal 25/50/75%",
)


# %% [markdown]
# ## 1. Historical data and the long-run yield trend
#
# The calibration panel covers 1996–2025. The first model is:
#
# \[
# TrendYield_t = \beta_0 + \beta_1(Year_t - 1996).
# \]
#
# Centering the year makes the intercept interpretable as the 1996 trend yield.
# Residuals are always defined as `actual − fitted`.

# %%
REQUIRED_PANEL_COLUMNS = (
    "year",
    "yield_bu_per_acre",
    "may_july_precip_inches",
    "may_july_avg_temp_f",
    "july_pdsi",
    "preseason_futures_usd_per_bushel",
    "july_update_futures_usd_per_bushel",
    "harvest_futures_usd_per_bushel",
)


def load_calibration_panel(path: Path, config: ModelConfig) -> pd.DataFrame:
    """Load and validate the 30-year source-backed calibration panel."""

    panel = pd.read_csv(path)
    missing = sorted(set(REQUIRED_PANEL_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"Calibration panel is missing columns: {missing}")

    panel = panel.sort_values("year").reset_index(drop=True)
    expected_years = list(
        range(config.calibration_start_year, config.calibration_end_year + 1)
    )
    if panel["year"].tolist() != expected_years:
        raise ValueError(
            "Calibration years must be consecutive and exactly match the configured range."
        )
    if panel[list(REQUIRED_PANEL_COLUMNS)].isna().any().any():
        raise ValueError("Calibration panel contains missing required values.")
    numeric = panel[list(REQUIRED_PANEL_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Calibration panel contains non-finite required values.")
    if (panel["yield_bu_per_acre"] < 0).any():
        raise ValueError("Historical yield cannot be negative.")
    price_columns = [
        "preseason_futures_usd_per_bushel",
        "july_update_futures_usd_per_bushel",
        "harvest_futures_usd_per_bushel",
    ]
    if (panel[price_columns] <= 0).any().any():
        raise ValueError("Historical futures prices must be positive.")

    panel = panel.copy()
    panel["trend_index"] = panel["year"] - config.calibration_start_year
    panel["july_futures_change"] = (
        panel["july_update_futures_usd_per_bushel"]
        - panel["preseason_futures_usd_per_bushel"]
    )
    panel["harvest_futures_change"] = (
        panel["harvest_futures_usd_per_bushel"]
        - panel["preseason_futures_usd_per_bushel"]
    )
    return panel


@dataclass(frozen=True)
class OlsFit:
    """Small, dependency-light OLS result used throughout the framework."""

    name: str
    outcome: str
    predictors: tuple[str, ...]
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    residual_standard_deviation: float
    loocv_rmse: float

    def coefficient(self, term: str) -> float:
        names = ("intercept", *self.predictors)
        if term not in names:
            raise KeyError(f"Unknown coefficient {term!r}; available terms: {names}")
        return float(self.coefficients[names.index(term)])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = np.column_stack(
            [
                np.ones(len(frame)),
                *[frame[column].to_numpy(dtype=float) for column in self.predictors],
            ]
        )
        return design @ self.coefficients


def _loocv_rmse(frame: pd.DataFrame, predictors: Iterable[str], outcome: str) -> float:
    predictors = tuple(predictors)
    errors: list[float] = []
    for test_index in range(len(frame)):
        train_mask = np.arange(len(frame)) != test_index
        train = frame.loc[train_mask]
        design_train = np.column_stack(
            [
                np.ones(len(train)),
                *[train[column].to_numpy(dtype=float) for column in predictors],
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(
            design_train,
            train[outcome].to_numpy(dtype=float),
            rcond=None,
        )
        row = frame.iloc[test_index]
        design_test = np.array(
            [1.0, *[float(row[column]) for column in predictors]], dtype=float
        )
        prediction = float(design_test @ beta)
        errors.append(float(row[outcome]) - prediction)
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_ols(
    frame: pd.DataFrame,
    predictors: Iterable[str],
    outcome: str,
    name: str,
) -> OlsFit:
    """Fit OLS with an intercept and report in-sample and LOOCV diagnostics."""

    predictors = tuple(predictors)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            *[frame[column].to_numpy(dtype=float) for column in predictors],
        ]
    )
    target = frame[outcome].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != design.shape[1]:
        raise ValueError(f"Design matrix is rank-deficient for model {name!r}.")

    fitted = design @ coefficients
    residuals = target - fitted
    n_observations = len(frame)
    n_parameters = design.shape[1]
    sse = float(np.sum(np.square(residuals)))
    sst = float(np.sum(np.square(target - target.mean())))
    r_squared = 1.0 - sse / sst
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (
        (n_observations - 1) / (n_observations - n_parameters)
    )
    rmse = float(np.sqrt(sse / n_observations))
    residual_sd = float(np.sqrt(sse / (n_observations - n_parameters)))

    return OlsFit(
        name=name,
        outcome=outcome,
        predictors=predictors,
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        rmse=rmse,
        residual_standard_deviation=residual_sd,
        loocv_rmse=_loocv_rmse(frame, predictors, outcome),
    )


# %% [markdown]
# ## 2. July PDSI yield model and model-selection discipline
#
# The selected July model is:
#
# \[
# Yield_t = \beta_0 + \beta_1 TrendIndex_t + \beta_2 JulyPDSI_t + \epsilon_t.
# \]
#
# PDSI is interpreted as a historical association, not a causal estimate. A
# higher PDSI generally means wetter conditions. To avoid selecting a model
# merely because it fits the 30 observations well, the framework compares
# leave-one-out cross-validation RMSE. The eligible set is trend-only plus each
# trend-plus-one-weather-variable model; larger weather combinations are kept as
# robustness diagnostics rather than allowed to win the primary selection.

# %%
@dataclass(frozen=True)
class CalibrationResults:
    panel: pd.DataFrame
    diagnostics: pd.DataFrame
    trend_yield_model: OlsFit
    selected_yield_model: OlsFit
    july_price_model: OlsFit
    harvest_price_model: OlsFit
    shock_library: pd.DataFrame
    paired_price_residual_correlation: float


def _diagnostic_row(
    section: str,
    fit: OlsFit,
    eligible: bool,
    selected: bool,
) -> dict[str, Any]:
    return {
        "section": section,
        "model": fit.name,
        "predictors": " + ".join(fit.predictors),
        "eligible_for_selection": eligible,
        "n_observations": len(fit.fitted),
        "n_estimated_parameters": len(fit.coefficients),
        "r_squared": fit.r_squared,
        "adjusted_r_squared": fit.adjusted_r_squared,
        "in_sample_rmse": fit.rmse,
        "residual_standard_deviation": fit.residual_standard_deviation,
        "loocv_rmse": fit.loocv_rmse,
        "selected": selected,
    }


def calibrate_models(panel: pd.DataFrame) -> CalibrationResults:
    """Re-estimate every model used in Lessons 1–5 from the historical panel."""

    yield_candidates: list[tuple[str, tuple[str, ...], bool]] = [
        ("trend_only", ("trend_index",), True),
        ("trend_plus_precip", ("trend_index", "may_july_precip_inches"), True),
        ("trend_plus_temperature", ("trend_index", "may_july_avg_temp_f"), True),
        ("trend_plus_pdsi", ("trend_index", "july_pdsi"), True),
        (
            "trend_plus_precip_temperature",
            ("trend_index", "may_july_precip_inches", "may_july_avg_temp_f"),
            False,
        ),
        (
            "trend_plus_precip_pdsi",
            ("trend_index", "may_july_precip_inches", "july_pdsi"),
            False,
        ),
        (
            "trend_plus_temperature_pdsi",
            ("trend_index", "may_july_avg_temp_f", "july_pdsi"),
            False,
        ),
        (
            "trend_plus_all_weather",
            (
                "trend_index",
                "may_july_precip_inches",
                "may_july_avg_temp_f",
                "july_pdsi",
            ),
            False,
        ),
    ]

    yield_fits: dict[str, OlsFit] = {}
    eligibility: dict[str, bool] = {}
    for name, predictors, eligible in yield_candidates:
        yield_fits[name] = fit_ols(
            panel,
            predictors,
            outcome="yield_bu_per_acre",
            name=name,
        )
        eligibility[name] = eligible

    eligible_names = [name for name, _, eligible in yield_candidates if eligible]
    selected_name = min(
        eligible_names,
        key=lambda model_name: yield_fits[model_name].loocv_rmse,
    )
    if selected_name != "trend_plus_pdsi":
        raise AssertionError(
            f"Expected trend_plus_pdsi to win the pre-specified selection; got {selected_name}."
        )

    trend_fit = yield_fits["trend_only"]
    selected_yield_fit = yield_fits[selected_name]
    calibrated = panel.copy()
    calibrated["trend_yield"] = trend_fit.fitted
    calibrated["july_yield_forecast"] = selected_yield_fit.fitted
    calibrated["july_weather_yield_signal"] = (
        calibrated["july_yield_forecast"] - calibrated["trend_yield"]
    )
    calibrated["final_yield_surprise_vs_trend"] = (
        calibrated["yield_bu_per_acre"] - calibrated["trend_yield"]
    )

    july_price_fit = fit_ols(
        calibrated,
        ("july_weather_yield_signal",),
        outcome="july_futures_change",
        name="july_futures_change",
    )
    harvest_price_fit = fit_ols(
        calibrated,
        ("final_yield_surprise_vs_trend",),
        outcome="harvest_futures_change",
        name="harvest_futures_change",
    )

    calibrated["yield_residual"] = selected_yield_fit.residuals
    calibrated["july_price_residual"] = july_price_fit.residuals
    calibrated["harvest_price_residual"] = harvest_price_fit.residuals

    shock_library = calibrated[
        [
            "year",
            "may_july_precip_inches",
            "may_july_avg_temp_f",
            "july_pdsi",
            "yield_residual",
            "july_price_residual",
            "harvest_price_residual",
        ]
    ].copy()
    paired_correlation = float(
        shock_library[["july_price_residual", "harvest_price_residual"]]
        .corr()
        .iloc[0, 1]
    )

    diagnostic_rows = [
        _diagnostic_row(
            "yield_model",
            yield_fits[name],
            eligibility[name],
            name == selected_name,
        )
        for name, _, _ in yield_candidates
    ]
    diagnostic_rows.extend(
        [
            _diagnostic_row("price_model", july_price_fit, True, True),
            _diagnostic_row("price_model", harvest_price_fit, True, True),
        ]
    )

    return CalibrationResults(
        panel=calibrated,
        diagnostics=pd.DataFrame(diagnostic_rows),
        trend_yield_model=trend_fit,
        selected_yield_model=selected_yield_fit,
        july_price_model=july_price_fit,
        harvest_price_model=harvest_price_fit,
        shock_library=shock_library,
        paired_price_residual_correlation=paired_correlation,
    )


# %% [markdown]
# ## 3. Historical residual bootstrap for yield
#
# July PDSI and the selected yield-model residual are sampled independently from
# their historical empirical distributions. The July forecast uses PDSI; final
# yield adds a residual only after the July decision:
#
# \[
# \widehat{Y}_{July,i}=\beta_0+\beta_1(30)+\beta_2PDSI_i
# \]
#
# \[
# Y_i=\max(0,\widehat{Y}_{July,i}+e^Y_i).
# \]
#
# Independent weather and yield-residual draws are an explicit modeling choice.
# They allow new combinations of historically observed shocks, but they do not
# claim those components are structurally independent in reality.

# %% [markdown]
# ## 4. Futures-price calibration
#
# Price changes are measured from the March preseason futures price, not from
# one observation date to the next:
#
# \[
# \Delta F_{July}=\alpha_J+\gamma_J(JulyYieldForecast-TrendYield)+e^J
# \]
#
# \[
# \Delta F_{Harvest}=\alpha_H+\gamma_H(FinalYield-TrendYield)+e^H.
# \]
#
# July and harvest price residuals are sampled as a same-year pair, preserving
# their historical dependence. The July model is intentionally described as
# weak: its R² is about 0.022, so Lesson 11 removes its yield-signal coefficient
# in a robustness test.

# %% [markdown]
# ## 5. Joint simulation of yield, futures prices, basis, and cash price
#
# Random draws follow the exact lesson sequence:
#
# 1. historical July PDSI source year;
# 2. historical yield-residual source year;
# 3. same-year July/harvest price-residual pair;
# 4. a uniform draw transformed through the triangular basis inverse CDF.
#
# July and harvest futures changes are both cumulative changes from the same
# March price. Harvest change is **not** added on top of the July price.

# %%
def _predict_at(fit: OlsFit, values: dict[str, float | np.ndarray]) -> np.ndarray:
    lengths = [np.asarray(value).size for value in values.values() if np.asarray(value).ndim]
    n = max(lengths, default=1)
    design_columns: list[np.ndarray] = [np.ones(n)]
    for predictor in fit.predictors:
        value = np.asarray(values[predictor], dtype=float)
        if value.ndim == 0:
            value = np.full(n, float(value))
        design_columns.append(value)
    return np.column_stack(design_columns) @ fit.coefficients


def inverse_triangular_cdf(
    uniform_draw: np.ndarray,
    low: float,
    mode: float,
    high: float,
) -> np.ndarray:
    """Transparent inverse-CDF implementation used by Lesson 6 and Excel."""

    uniform_draw = np.asarray(uniform_draw, dtype=float)
    if ((uniform_draw < 0) | (uniform_draw >= 1)).any():
        raise ValueError("Triangular inverse-CDF draws must lie in [0, 1).")
    cutoff = (mode - low) / (high - low)
    return np.where(
        uniform_draw < cutoff,
        low + np.sqrt(uniform_draw * (high - low) * (mode - low)),
        high
        - np.sqrt((1.0 - uniform_draw) * (high - low) * (high - mode)),
    )


def simulate_common_scenarios(
    calibration: CalibrationResults,
    config: ModelConfig,
) -> pd.DataFrame:
    """Generate the common scenarios used by every hedge strategy."""

    shocks = calibration.shock_library.reset_index(drop=True)
    n_history = len(shocks)
    rng = np.random.default_rng(config.random_seed)

    weather_rows = rng.integers(0, n_history, size=config.n_simulations)
    yield_rows = rng.integers(0, n_history, size=config.n_simulations)
    price_rows = rng.integers(0, n_history, size=config.n_simulations)
    basis_uniform = rng.random(config.n_simulations)

    weather_source_year = shocks["year"].to_numpy()[weather_rows]
    yield_residual_source_year = shocks["year"].to_numpy()[yield_rows]
    price_residual_source_year = shocks["year"].to_numpy()[price_rows]
    july_pdsi = shocks["july_pdsi"].to_numpy()[weather_rows]
    yield_residual = shocks["yield_residual"].to_numpy()[yield_rows]
    july_price_residual = shocks["july_price_residual"].to_numpy()[price_rows]
    harvest_price_residual = shocks["harvest_price_residual"].to_numpy()[price_rows]

    trend_index = config.forecast_trend_index
    trend_yield = float(
        _predict_at(calibration.trend_yield_model, {"trend_index": trend_index})[0]
    )
    july_yield_forecast = _predict_at(
        calibration.selected_yield_model,
        {"trend_index": trend_index, "july_pdsi": july_pdsi},
    )
    final_yield = np.maximum(july_yield_forecast + yield_residual, 0.0)

    july_yield_signal = july_yield_forecast - trend_yield
    final_yield_surprise = final_yield - trend_yield
    july_futures_change = (
        calibration.july_price_model.coefficient("intercept")
        + calibration.july_price_model.coefficient("july_weather_yield_signal")
        * july_yield_signal
        + july_price_residual
    )
    harvest_futures_change = (
        calibration.harvest_price_model.coefficient("intercept")
        + calibration.harvest_price_model.coefficient(
            "final_yield_surprise_vs_trend"
        )
        * final_yield_surprise
        + harvest_price_residual
    )
    july_futures = np.maximum(
        config.futures_price_floor_usd_per_bushel,
        config.preseason_futures_usd_per_bushel + july_futures_change,
    )
    harvest_futures = np.maximum(
        config.futures_price_floor_usd_per_bushel,
        config.preseason_futures_usd_per_bushel + harvest_futures_change,
    )

    basis = inverse_triangular_cdf(
        basis_uniform,
        config.basis_low_usd_per_bushel,
        config.basis_mode_usd_per_bushel,
        config.basis_high_usd_per_bushel,
    )
    raw_cash_price = harvest_futures + basis
    cash_price = np.maximum(config.cash_price_floor_usd_per_bushel, raw_cash_price)
    actual_production = final_yield * config.farm_acres
    cash_revenue = actual_production * cash_price
    unhedged_profit = cash_revenue - config.total_production_cost_usd

    scenarios = pd.DataFrame(
        {
            "scenario_id": np.arange(1, config.n_simulations + 1),
            "weather_source_year": weather_source_year,
            "yield_residual_source_year": yield_residual_source_year,
            "price_residual_source_year": price_residual_source_year,
            "july_pdsi": july_pdsi,
            "yield_residual_bu_per_acre": yield_residual,
            "july_price_residual_usd_per_bushel": july_price_residual,
            "harvest_price_residual_usd_per_bushel": harvest_price_residual,
            "preseason_expected_yield_bu_per_acre": trend_yield,
            "july_yield_forecast_bu_per_acre": july_yield_forecast,
            "final_yield_bu_per_acre": final_yield,
            "july_yield_signal_bu_per_acre": july_yield_signal,
            "final_yield_surprise_bu_per_acre": final_yield_surprise,
            "preseason_futures_usd_per_bushel": config.preseason_futures_usd_per_bushel,
            "july_futures_usd_per_bushel": july_futures,
            "harvest_futures_usd_per_bushel": harvest_futures,
            "basis_uniform_draw": basis_uniform,
            "basis_usd_per_bushel": basis,
            "raw_cash_price_usd_per_bushel": raw_cash_price,
            "cash_price_usd_per_bushel": cash_price,
            "actual_production_bushels": actual_production,
            "cash_revenue_usd": cash_revenue,
            "production_cost_usd": config.total_production_cost_usd,
            "unhedged_profit_usd": unhedged_profit,
            "unhedged_profit_usd_per_acre": unhedged_profit / config.farm_acres,
        }
    )
    return scenarios


# %% [markdown]
# ## 6. Basis and harvest cash price
#
# Basis follows Iowa State's definition:
#
# \[
# Basis = CashPrice - FuturesPrice,
# \qquad CashPrice = HarvestFutures + Basis.
# \]
#
# The triangular distribution uses low = −$0.28/bu, mode = −$0.20/bu, and
# high = −$0.11/bu. Treating the five-year average basis as the mode and drawing
# basis independently from futures are transparent simplifying assumptions.

# %% [markdown]
# ## 7. Unhedged farm-profit baseline
#
# For the representative 1,000-acre farm:
#
# \[
# Production_i=Yield_i\times Acres,
# \]
#
# \[
# CashRevenue_i=Production_i\times CashPrice_i,
# \]
#
# \[
# UnhedgedProfit_i=CashRevenue_i-(911.98\times Acres).
# \]
#
# Production cost is held fixed per acre. Crop insurance, government payments,
# taxes, quality discounts, storage decisions, and yield-varying costs are not
# modeled and should remain explicit limitations.

# %% [markdown]
# ## 8. Fixed 50% hedge and two-lot futures accounting
#
# Positive contract counts are short futures positions. The framework keeps the
# March lot and every July adjustment separate:
#
# \[
# InitialPnL_i=N_{0,i}\times5{,}000\times(F_0-F_{H,i}),
# \]
#
# \[
# JulyAdjustmentPnL_i=\Delta N_i\times5{,}000\times(F_{July,i}-F_{H,i}).
# \]
#
# A negative July adjustment means buying back part of the original short; the
# signed formula remains correct. Fixed strategies have a zero July adjustment,
# but the zero is still shown explicitly for auditability.

# %%
def rounded_contracts(quantity_bushels: float | np.ndarray, contract_size: int) -> np.ndarray:
    """Nearest-whole-contract rounding for nonnegative quantities."""

    quantity = np.maximum(np.asarray(quantity_bushels, dtype=float), 0.0)
    return np.floor(quantity / contract_size + 0.5).astype(int)


def strategy_positions(
    strategy: str,
    scenarios: pd.DataFrame,
    config: ModelConfig,
    pdsi_low_threshold: float,
    pdsi_high_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial contracts, July adjustment, and July target ratio."""

    expected_production = (
        scenarios["preseason_expected_yield_bu_per_acre"].to_numpy()
        * config.farm_acres
    )
    updated_production = (
        scenarios["july_yield_forecast_bu_per_acre"].to_numpy()
        * config.farm_acres
    )
    n_scenarios = len(scenarios)

    if strategy.startswith("Fixed "):
        ratio = float(strategy.split()[1].rstrip("%")) / 100.0
        initial = rounded_contracts(
            ratio * expected_production, config.contract_size_bushels
        )
        adjustment = np.zeros(n_scenarios, dtype=int)
        july_ratio = np.full(n_scenarios, ratio)
        return initial, adjustment, july_ratio

    initial = rounded_contracts(
        config.adaptive_initial_ratio * expected_production,
        config.contract_size_bushels,
    )
    if strategy == "Updated 50%":
        july_ratio = np.full(n_scenarios, config.normal_target_ratio)
    elif strategy == "Weather-signal 25/50/75%":
        pdsi = scenarios["july_pdsi"].to_numpy()
        july_ratio = np.select(
            [pdsi < pdsi_low_threshold, pdsi > pdsi_high_threshold],
            [config.drought_target_ratio, config.wet_target_ratio],
            default=config.normal_target_ratio,
        ).astype(float)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    final_contracts = rounded_contracts(
        july_ratio * updated_production, config.contract_size_bushels
    )
    adjustment = final_contracts - initial
    return initial, adjustment, july_ratio


def evaluate_strategy(
    strategy: str,
    scenarios: pd.DataFrame,
    config: ModelConfig,
    pdsi_low_threshold: float,
    pdsi_high_threshold: float,
) -> pd.DataFrame:
    """Apply one position rule and one common P&L engine to every scenario."""

    initial, adjustment, july_ratio = strategy_positions(
        strategy,
        scenarios,
        config,
        pdsi_low_threshold,
        pdsi_high_threshold,
    )
    final_contracts = initial + adjustment
    contract_size = config.contract_size_bushels
    f0 = scenarios["preseason_futures_usd_per_bushel"].to_numpy()
    f_july = scenarios["july_futures_usd_per_bushel"].to_numpy()
    f_harvest = scenarios["harvest_futures_usd_per_bushel"].to_numpy()

    initial_pnl = initial * contract_size * (f0 - f_harvest)
    july_adjustment_pnl = adjustment * contract_size * (f_july - f_harvest)
    total_futures_pnl = initial_pnl + july_adjustment_pnl

    # Independent interval identity: this catches transaction-price mistakes.
    preseason_to_july_pnl = initial * contract_size * (f0 - f_july)
    july_to_harvest_pnl = final_contracts * contract_size * (f_july - f_harvest)
    if not np.allclose(
        total_futures_pnl,
        preseason_to_july_pnl + july_to_harvest_pnl,
        atol=1e-8,
    ):
        raise AssertionError("Lot-based and interval-based futures P&L do not agree.")

    transaction_sides = (
        np.abs(initial) + np.abs(adjustment) + np.abs(final_contracts)
    )
    transaction_cost = (
        transaction_sides * config.transaction_cost_usd_per_contract_side
    )
    margin_financing_cost = (
        config.initial_margin_usd_per_contract
        * config.annual_margin_financing_rate
        * (
            np.abs(initial) * config.days_preseason_to_july / config.days_per_year
            + np.abs(final_contracts)
            * config.days_july_to_harvest
            / config.days_per_year
        )
    )

    cash_revenue = scenarios["cash_revenue_usd"].to_numpy()
    gross_revenue_after_hedge = (
        cash_revenue + total_futures_pnl - transaction_cost - margin_financing_cost
    )
    profit = gross_revenue_after_hedge - scenarios["production_cost_usd"].to_numpy()

    actual_production = scenarios["actual_production_bushels"].to_numpy()
    final_hedged_bushels = final_contracts * contract_size
    overhedged = final_hedged_bushels > actual_production

    # This is a two-date liquidity proxy, not a daily broker margin ledger.
    july_mark_to_market = initial * contract_size * (f0 - f_july)
    harvest_segment_mark_to_market = (
        final_contracts * contract_size * (f_july - f_harvest)
    )
    margin_call_proxy = (
        (-july_mark_to_market > config.margin_liquidity_reserve_usd)
        | (
            -harvest_segment_mark_to_market
            > config.margin_liquidity_reserve_usd
        )
    )

    initial_ratio = (
        float(strategy.split()[1].rstrip("%")) / 100.0
        if strategy.startswith("Fixed ")
        else config.adaptive_initial_ratio
    )
    return pd.DataFrame(
        {
            "scenario_id": scenarios["scenario_id"].to_numpy(),
            "strategy": strategy,
            "initial_hedge_ratio": initial_ratio,
            "july_target_hedge_ratio": july_ratio,
            "initial_contracts": initial,
            "july_adjustment_contracts": adjustment,
            "final_contracts": final_contracts,
            "actual_production_bushels": actual_production,
            "final_hedged_bushels": final_hedged_bushels,
            "cash_revenue_usd": cash_revenue,
            "initial_futures_pnl_usd": initial_pnl,
            "july_adjustment_pnl_usd": july_adjustment_pnl,
            "total_futures_pnl_usd": total_futures_pnl,
            "transaction_cost_usd": transaction_cost,
            "margin_financing_cost_usd": margin_financing_cost,
            "gross_revenue_after_hedge_usd": gross_revenue_after_hedge,
            "profit_usd": profit,
            "profit_usd_per_acre": profit / config.farm_acres,
            "overhedged": overhedged,
            "margin_call_proxy": margin_call_proxy,
        }
    )


# %% [markdown]
# ## 9. Compare fixed 0%, 25%, 50%, 75%, and 100% hedges
#
# Contract counts are rounded to the nearest whole 5,000-bushel contract with
# `floor(x + 0.5)`. This avoids Python's banker's rounding and matches Excel.
# Common random numbers make each scenario-level profit difference attributable
# to the strategy rule rather than to different weather or price draws.
#
# A 100% target is not automatically safest: if actual production falls below
# the pre-harvest short position, the farm becomes over-hedged and reintroduces
# price exposure.

# %% [markdown]
# ## 10. July-updated strategies
#
# Both dynamic policies begin at 50% in March.
#
# - **Updated 50%:** remain at 50%, but recompute bushels using the July yield
#   forecast.
# - **Weather-signal 25/50/75%:** target 25% below the historical lower PDSI
#   tercile, 50% between terciles, and 75% above the upper tercile; apply the
#   chosen ratio to the July yield forecast.
#
# The terciles are estimated from the 1996–2025 July PDSI sample. The rule is a
# project policy, not an exchange or extension-service recommendation.

# %%
def evaluate_all_strategies(
    scenarios: pd.DataFrame,
    config: ModelConfig,
    pdsi_low_threshold: float,
    pdsi_high_threshold: float,
) -> pd.DataFrame:
    return pd.concat(
        [
            evaluate_strategy(
                strategy,
                scenarios,
                config,
                pdsi_low_threshold,
                pdsi_high_threshold,
            )
            for strategy in ALL_STRATEGIES
        ],
        ignore_index=True,
    )


def summarize_strategies(results: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Report mean, variability, lower-tail risk, and implementation risk."""

    rows: list[dict[str, Any]] = []
    for strategy, group in results.groupby("strategy", sort=False):
        profit = group["profit_usd_per_acre"]
        p5 = float(profit.quantile(config.lower_tail_probability))
        cvar5 = float(profit.loc[profit <= p5].mean())
        standard_error = float(profit.std(ddof=1) / np.sqrt(len(profit)))
        rows.append(
            {
                "strategy": strategy,
                "Expected Gross Revenue ($/acre)": (
                    group["gross_revenue_after_hedge_usd"].mean()
                    / config.farm_acres
                ),
                "Expected Profit ($/acre)": profit.mean(),
                "Expected Profit (farm $)": group["profit_usd"].mean(),
                "Profit Std Dev ($/acre)": profit.std(ddof=1),
                "Probability Profit < 0": (profit < 0).mean(),
                "P5 Profit ($/acre)": p5,
                "CVaR 5% Profit ($/acre)": cvar5,
                "Probability Overhedged": group["overhedged"].mean(),
                "Probability Margin Call Proxy": group["margin_call_proxy"].mean(),
                "Avg Transaction Cost ($/acre)": (
                    group["transaction_cost_usd"].mean() / config.farm_acres
                ),
                "Avg Margin Financing Cost ($/acre)": (
                    group["margin_financing_cost_usd"].mean() / config.farm_acres
                ),
                "Mean 95% CI Low ($/acre)": (
                    profit.mean() - config.z_95 * standard_error
                ),
                "Mean 95% CI High ($/acre)": (
                    profit.mean() + config.z_95 * standard_error
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class StrategyDecision:
    preferred_strategy: str
    eligible_strategies: tuple[str, ...]
    unhedged_mean_profit_usd_per_acre: float
    minimum_eligible_mean_profit_usd_per_acre: float
    maximum_overhedge_probability: float
    primary_objective: str
    expected_profit_constraint: str
    overhedge_constraint: str
    tie_breaker: str


def choose_preferred_strategy(
    summary: pd.DataFrame,
    config: ModelConfig,
) -> tuple[pd.DataFrame, StrategyDecision]:
    """Apply the pre-specified eligibility filters, then maximize 5% CVaR."""

    scored = summary.copy()
    unhedged = scored.loc[scored["strategy"] == "Fixed 0%"]
    if len(unhedged) != 1:
        raise ValueError("Exactly one Fixed 0% row is required for selection.")
    unhedged_mean = float(unhedged.iloc[0]["Expected Profit ($/acre)"])
    minimum_mean = unhedged_mean - (
        config.max_mean_profit_shortfall_fraction * abs(unhedged_mean)
    )
    scored["Pass Expected-Profit Rule"] = (
        scored["Expected Profit ($/acre)"] >= minimum_mean
    )
    scored["Pass Overhedge Rule"] = (
        scored["Probability Overhedged"] <= config.max_overhedge_probability
    )
    scored["Eligible"] = (
        scored["Pass Expected-Profit Rule"] & scored["Pass Overhedge Rule"]
    )
    eligible = scored.loc[scored["Eligible"]].sort_values(
        ["CVaR 5% Profit ($/acre)", "Expected Profit ($/acre)"],
        ascending=[False, False],
    )
    if eligible.empty:
        raise ValueError("No strategy passes the pre-specified safeguards.")
    preferred = str(eligible.iloc[0]["strategy"])
    scored["Preferred"] = scored["strategy"].eq(preferred)

    decision = StrategyDecision(
        preferred_strategy=preferred,
        eligible_strategies=tuple(eligible["strategy"]),
        unhedged_mean_profit_usd_per_acre=unhedged_mean,
        minimum_eligible_mean_profit_usd_per_acre=minimum_mean,
        maximum_overhedge_probability=config.max_overhedge_probability,
        primary_objective=(
            "Maximize lower-tail 5% profit CVaR among eligible strategies."
        ),
        expected_profit_constraint=(
            "Mean profit cannot be more than 5% of the absolute unhedged mean "
            "below the unhedged mean."
        ),
        overhedge_constraint="Over-hedging probability cannot exceed 10%.",
        tie_breaker="If CVaR ties, choose the higher expected profit.",
    )
    return scored, decision


# %% [markdown]
# ## 11. Final comparison and July-price robustness test
#
# The July price regression has little explanatory power. The robustness case
# therefore holds the same random draws, yield, harvest futures, basis, and cash
# revenue fixed, but sets only the July yield-signal slope to zero:
#
# \[
# F^{robust}_{July}=\max(1.50,F_0+\alpha_J+e^J).
# \]
#
# Fixed strategies do not rebalance in July, so their final profit is unchanged.
# Updated policies can change because their July adjustment uses the July price.
# The decision is robust only if the preferred strategy remains the same.

# %%
def build_july_beta_zero_scenarios(
    scenarios: pd.DataFrame,
    calibration: CalibrationResults,
    config: ModelConfig,
) -> pd.DataFrame:
    robust = scenarios.copy()
    robust["july_futures_usd_per_bushel"] = np.maximum(
        config.futures_price_floor_usd_per_bushel,
        robust["preseason_futures_usd_per_bushel"].to_numpy()
        + calibration.july_price_model.coefficient("intercept")
        + robust["july_price_residual_usd_per_bushel"].to_numpy(),
    )
    return robust


@dataclass(frozen=True)
class MasterFrameworkResults:
    config: ModelConfig
    calibration: CalibrationResults
    scenarios: pd.DataFrame
    main_strategy_results: pd.DataFrame
    main_summary: pd.DataFrame
    main_decision: StrategyDecision
    robustness_scenarios: pd.DataFrame
    robustness_strategy_results: pd.DataFrame
    robustness_summary: pd.DataFrame
    robustness_decision: StrategyDecision
    diagnostics: dict[str, Any]


def _distribution_summary(series: pd.Series | np.ndarray) -> dict[str, float]:
    values = pd.Series(series, dtype=float)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "p5": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "maximum": float(values.max()),
    }


def validate_results(results: MasterFrameworkResults) -> None:
    """Run structural, accounting, and benchmark reconciliation checks."""

    config = results.config
    calibration = results.calibration
    scenarios = results.scenarios
    main = results.main_strategy_results
    robust = results.robustness_strategy_results

    if len(calibration.panel) != 30:
        raise AssertionError("Expected exactly 30 calibration years.")
    if len(scenarios) != config.n_simulations or not scenarios["scenario_id"].is_unique:
        raise AssertionError("Scenario count or identifier uniqueness check failed.")
    if scenarios.isna().any().any():
        raise AssertionError("Common scenarios contain missing values.")
    if (scenarios["final_yield_bu_per_acre"] < 0).any():
        raise AssertionError("A simulated yield is negative.")
    if (
        scenarios["july_futures_usd_per_bushel"]
        < config.futures_price_floor_usd_per_bushel
    ).any() or (
        scenarios["harvest_futures_usd_per_bushel"]
        < config.futures_price_floor_usd_per_bushel
    ).any():
        raise AssertionError("A simulated futures price violates the price floor.")
    if (
        scenarios["cash_price_usd_per_bushel"]
        < config.cash_price_floor_usd_per_bushel
    ).any():
        raise AssertionError("A simulated cash price violates the cash-price floor.")

    expected_rows = config.n_simulations * len(ALL_STRATEGIES)
    for label, frame in (("main", main), ("robustness", robust)):
        if len(frame) != expected_rows:
            raise AssertionError(f"{label} strategy results have the wrong row count.")
        if frame.duplicated(["scenario_id", "strategy"]).any():
            raise AssertionError(f"{label} results contain duplicate scenario-strategy rows.")
        if frame.isna().any().any():
            raise AssertionError(f"{label} results contain missing values.")
        fixed = frame[frame["strategy"].str.startswith("Fixed ")]
        if (fixed["july_adjustment_contracts"] != 0).any():
            raise AssertionError("A fixed strategy changed contracts in July.")

    if results.main_summary["Preferred"].sum() != 1:
        raise AssertionError("Main results must have exactly one preferred strategy.")
    if results.robustness_summary["Preferred"].sum() != 1:
        raise AssertionError("Robustness results must have exactly one preferred strategy.")

    # Full benchmark checks apply only to the locked 10,000-trial lesson run.
    if config.n_simulations == 10_000 and config.random_seed == 8_122_026:
        coefficient_checks = {
            "trend_intercept": (
                calibration.trend_yield_model.coefficient("intercept"),
                140.9268817204301,
            ),
            "trend_slope": (
                calibration.trend_yield_model.coefficient("trend_index"),
                2.310789766407122,
            ),
            "yield_intercept": (
                calibration.selected_yield_model.coefficient("intercept"),
                138.3603607964906,
            ),
            "yield_trend": (
                calibration.selected_yield_model.coefficient("trend_index"),
                2.318714127208453,
            ),
            "yield_pdsi": (
                calibration.selected_yield_model.coefficient("july_pdsi"),
                1.907873690521471,
            ),
            "july_price_intercept": (
                calibration.july_price_model.coefficient("intercept"),
                -0.05375000000000204,
            ),
            "july_price_slope": (
                calibration.july_price_model.coefficient(
                    "july_weather_yield_signal"
                ),
                -0.019741301483521958,
            ),
            "harvest_price_intercept": (
                calibration.harvest_price_model.coefficient("intercept"),
                -0.1506666666666673,
            ),
            "harvest_price_slope": (
                calibration.harvest_price_model.coefficient(
                    "final_yield_surprise_vs_trend"
                ),
                -0.026009894449960193,
            ),
            "paired_price_residual_correlation": (
                calibration.paired_price_residual_correlation,
                0.34724696594224497,
            ),
        }
        for label, (actual, expected) in coefficient_checks.items():
            if not np.isclose(actual, expected, atol=1e-10):
                raise AssertionError(f"{label}: {actual} != {expected}")

        scenario_checks = {
            "july_futures_mean": (
                scenarios["july_futures_usd_per_bushel"].mean(),
                4.637581762133069,
            ),
            "harvest_futures_mean": (
                scenarios["harvest_futures_usd_per_bushel"].mean(),
                4.541299446091132,
            ),
            "basis_mean": (
                scenarios["basis_usd_per_bushel"].mean(),
                -0.19706122942428791,
            ),
            "cash_price_mean": (
                scenarios["cash_price_usd_per_bushel"].mean(),
                4.344238216666843,
            ),
        }
        for label, (actual, expected) in scenario_checks.items():
            if not np.isclose(actual, expected, atol=1e-10):
                raise AssertionError(f"{label}: {actual} != {expected}")

        expected_main_metrics = {
            "Fixed 0%": (-1.5944606006, -338.7472297638, 0.0000),
            "Fixed 25%": (5.4810561656, -240.0045877430, 0.0000),
            "Fixed 50%": (11.9133441351, -154.8015607295, 0.0000),
            "Fixed 75%": (18.9888609010, -94.9466253290, 0.0000),
            "Fixed 100%": (25.4211488710, -118.1373768120, 0.4538),
            "Updated 50%": (12.0031094390, -155.0416200000, 0.0000),
            "Weather-signal 25/50/75%": (
                13.1318404060,
                -201.4749346266,
                0.0000,
            ),
        }
        for strategy, expected in expected_main_metrics.items():
            row = results.main_summary.loc[
                results.main_summary["strategy"] == strategy
            ].iloc[0]
            actual = (
                float(row["Expected Profit ($/acre)"]),
                float(row["CVaR 5% Profit ($/acre)"]),
                float(row["Probability Overhedged"]),
            )
            if not np.allclose(actual, expected, atol=1e-6):
                raise AssertionError(f"{strategy} benchmark mismatch: {actual} != {expected}")

        if results.main_decision.preferred_strategy != "Fixed 75%":
            raise AssertionError("Main benchmark recommendation must be Fixed 75%.")
        if results.robustness_decision.preferred_strategy != "Fixed 75%":
            raise AssertionError("Robustness benchmark recommendation must be Fixed 75%.")


def run_master_framework(
    project_root: str | Path,
    config: ModelConfig | None = None,
) -> MasterFrameworkResults:
    """Run calibration, simulation, strategy comparison, and robustness."""

    project_root = Path(project_root).resolve()
    config = config or ModelConfig()
    panel = load_calibration_panel(
        project_root / "processed" / "iowa_calibration_panel_1996_2025.csv",
        config,
    )
    calibration = calibrate_models(panel)
    scenarios = simulate_common_scenarios(calibration, config)
    low_threshold = float(calibration.panel["july_pdsi"].quantile(1 / 3))
    high_threshold = float(calibration.panel["july_pdsi"].quantile(2 / 3))

    main_results = evaluate_all_strategies(
        scenarios, config, low_threshold, high_threshold
    )
    main_summary, main_decision = choose_preferred_strategy(
        summarize_strategies(main_results, config), config
    )

    robustness_scenarios = build_july_beta_zero_scenarios(
        scenarios, calibration, config
    )
    robustness_results = evaluate_all_strategies(
        robustness_scenarios, config, low_threshold, high_threshold
    )
    robustness_summary, robustness_decision = choose_preferred_strategy(
        summarize_strategies(robustness_results, config), config
    )

    diagnostics = {
        "historical_years": len(panel),
        "scenario_count": len(scenarios),
        "strategy_count": len(ALL_STRATEGIES),
        "random_seed": config.random_seed,
        "weather_and_yield_residual_sampling": "independent historical rows",
        "price_residual_sampling": "same-year July/harvest pair",
        "paired_price_residual_correlation": (
            calibration.paired_price_residual_correlation
        ),
        "pdsi_lower_tercile": low_threshold,
        "pdsi_upper_tercile": high_threshold,
        "futures_price_floor_count_july": int(
            (
                scenarios["july_futures_usd_per_bushel"]
                <= config.futures_price_floor_usd_per_bushel + 1e-12
            ).sum()
        ),
        "futures_price_floor_count_harvest": int(
            (
                scenarios["harvest_futures_usd_per_bushel"]
                <= config.futures_price_floor_usd_per_bushel + 1e-12
            ).sum()
        ),
        "cash_price_floor_count": int(
            (
                scenarios["cash_price_usd_per_bushel"]
                <= config.cash_price_floor_usd_per_bushel + 1e-12
            ).sum()
        ),
        "main_preferred_strategy": main_decision.preferred_strategy,
        "robustness_preferred_strategy": robustness_decision.preferred_strategy,
        "recommendation_stable_when_july_beta_is_zero": (
            main_decision.preferred_strategy
            == robustness_decision.preferred_strategy
        ),
        "futures_pnl_identity": (
            "N0*Q*(F0-FH) + (NJ-N0)*Q*(FJuly-FH) = "
            "N0*Q*(F0-FJuly) + NJ*Q*(FJuly-FH)"
        ),
        "margin_call_limit": (
            "Two-date liquidity proxy at July and harvest; not a daily broker ledger."
        ),
    }

    results = MasterFrameworkResults(
        config=config,
        calibration=calibration,
        scenarios=scenarios,
        main_strategy_results=main_results,
        main_summary=main_summary,
        main_decision=main_decision,
        robustness_scenarios=robustness_scenarios,
        robustness_strategy_results=robustness_results,
        robustness_summary=robustness_summary,
        robustness_decision=robustness_decision,
        diagnostics=diagnostics,
    )
    validate_results(results)
    return results


# %% [markdown]
# ## Reporting and audit outputs
#
# The writer keeps compact summaries in ordinary CSV/JSON files and compresses
# the 10,000-scenario and 70,000 strategy-scenario tables. The chart emphasizes
# the distinction between average performance, lower-tail protection, and
# implementation risk.

# %%
def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_parameter_summary(results: MasterFrameworkResults) -> dict[str, Any]:
    calibration = results.calibration
    config = results.config
    return {
        "metadata": {
            "calibration_years": (
                f"{config.calibration_start_year}-{config.calibration_end_year}"
            ),
            "forecast_year": config.forecast_year,
            "historical_observations": len(calibration.panel),
        },
        "selected_yield_model": {
            "name": calibration.selected_yield_model.name,
            "coefficients": {
                "intercept": calibration.selected_yield_model.coefficient("intercept"),
                "trend_index": calibration.selected_yield_model.coefficient(
                    "trend_index"
                ),
                "july_pdsi": calibration.selected_yield_model.coefficient("july_pdsi"),
            },
            "adjusted_r_squared": calibration.selected_yield_model.adjusted_r_squared,
            "loocv_rmse_bu_per_acre": calibration.selected_yield_model.loocv_rmse,
            "interpretation": "Historical association, not a causal estimate.",
        },
        "trend_yield_model": {
            "intercept": calibration.trend_yield_model.coefficient("intercept"),
            "trend_index": calibration.trend_yield_model.coefficient("trend_index"),
        },
        "july_futures_change_model": {
            "intercept": calibration.july_price_model.coefficient("intercept"),
            "yield_signal_slope": calibration.july_price_model.coefficient(
                "july_weather_yield_signal"
            ),
            "r_squared": calibration.july_price_model.r_squared,
            "loocv_rmse_usd_per_bushel": calibration.july_price_model.loocv_rmse,
        },
        "harvest_futures_change_model": {
            "intercept": calibration.harvest_price_model.coefficient("intercept"),
            "yield_surprise_slope": calibration.harvest_price_model.coefficient(
                "final_yield_surprise_vs_trend"
            ),
            "r_squared": calibration.harvest_price_model.r_squared,
            "loocv_rmse_usd_per_bushel": calibration.harvest_price_model.loocv_rmse,
        },
        "operating_and_decision_assumptions": asdict(config),
    }


def plot_strategy_comparison(results: MasterFrameworkResults, path: Path) -> None:
    summary = results.main_summary
    x = np.arange(len(summary))
    labels = summary["strategy"].tolist()
    preferred = results.main_decision.preferred_strategy
    colors = ["#1F4E78" if label != preferred else "#70AD47" for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(x, summary["Expected Profit ($/acre)"], color=colors)
    axes[0, 0].axhline(0, color="#666666", linewidth=0.8)
    axes[0, 0].set_title("Expected Profit")
    axes[0, 0].set_ylabel("USD per acre")

    axes[0, 1].plot(
        x,
        summary["P5 Profit ($/acre)"],
        marker="o",
        label="5th percentile",
        color="#ED7D31",
    )
    axes[0, 1].plot(
        x,
        summary["CVaR 5% Profit ($/acre)"],
        marker="o",
        label="CVaR 5%",
        color="#C00000",
    )
    axes[0, 1].set_title("Lower-Tail Profit Protection")
    axes[0, 1].set_ylabel("USD per acre; higher is better")
    axes[0, 1].legend()

    axes[1, 0].bar(x, summary["Profit Std Dev ($/acre)"], color="#5B9BD5")
    axes[1, 0].set_title("Profit Variability")
    axes[1, 0].set_ylabel("USD per acre")

    width = 0.38
    axes[1, 1].bar(
        x - width / 2,
        summary["Probability Overhedged"],
        width,
        label="Over-hedged",
        color="#ED7D31",
    )
    axes[1, 1].bar(
        x + width / 2,
        summary["Probability Margin Call Proxy"],
        width,
        label="Margin-call proxy",
        color="#A5A5A5",
    )
    axes[1, 1].axhline(
        results.config.max_overhedge_probability,
        color="#C00000",
        linestyle="--",
        label="10% over-hedge limit",
    )
    axes[1, 1].set_title("Implementation Risks")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Iowa Corn Hedge Strategy Comparison — "
        f"{results.config.n_simulations:,} Common Scenarios",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(results: MasterFrameworkResults, path: Path) -> None:
    main = results.main_summary.set_index("strategy")
    robust = results.robustness_summary.set_index("strategy")
    labels = list(ALL_STRATEGIES)
    x = np.arange(len(labels))
    width = 0.38

    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(
        x - width / 2,
        main.loc[labels, "CVaR 5% Profit ($/acre)"],
        width,
        label="Main model",
        color="#4472C4",
    )
    axis.bar(
        x + width / 2,
        robust.loc[labels, "CVaR 5% Profit ($/acre)"],
        width,
        label="July price beta = 0",
        color="#70AD47",
    )
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylabel("CVaR 5% profit (USD per acre; higher is better)")
    axis.set_title("Robustness of the Downside-Risk Ranking")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    results: MasterFrameworkResults,
    output_directory: str | Path,
    save_trial_level_results: bool = True,
    create_plots: bool = True,
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    tables = output_directory / "tables"
    figures = output_directory / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    if create_plots:
        figures.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["calibration_diagnostics"] = tables / "calibration_diagnostics.csv"
    results.calibration.diagnostics.to_csv(
        paths["calibration_diagnostics"], index=False
    )
    paths["historical_shock_library"] = tables / "historical_shock_library.csv"
    results.calibration.shock_library.to_csv(
        paths["historical_shock_library"], index=False
    )
    paths["model_parameters"] = tables / "model_parameters.json"
    paths["model_parameters"].write_text(
        json.dumps(_json_ready(build_parameter_summary(results)), indent=2),
        encoding="utf-8",
    )

    paths["main_summary"] = tables / "main_strategy_summary.csv"
    results.main_summary.to_csv(paths["main_summary"], index=False)
    paths["robustness_summary"] = tables / "july_beta_zero_strategy_summary.csv"
    results.robustness_summary.to_csv(paths["robustness_summary"], index=False)

    paths["decision"] = tables / "final_decision.json"
    decision_payload = {
        "main_model": asdict(results.main_decision),
        "july_beta_zero_robustness": asdict(results.robustness_decision),
        "recommendation_unchanged": (
            results.main_decision.preferred_strategy
            == results.robustness_decision.preferred_strategy
        ),
        "correct_interpretation": (
            "Under the stated data, cost, basis, price-model, and implementation "
            "assumptions, Fixed 75% provides the strongest lower-tail profit "
            "protection among eligible strategies, and the recommendation is "
            "unchanged when the weak July price-signal coefficient is set to zero."
        ),
    }
    paths["decision"].write_text(
        json.dumps(_json_ready(decision_payload), indent=2), encoding="utf-8"
    )

    paths["diagnostics"] = tables / "framework_diagnostics.json"
    distribution_diagnostics = {
        **results.diagnostics,
        "yield_distribution_bu_per_acre": _distribution_summary(
            results.scenarios["final_yield_bu_per_acre"]
        ),
        "july_futures_distribution_usd_per_bushel": _distribution_summary(
            results.scenarios["july_futures_usd_per_bushel"]
        ),
        "harvest_futures_distribution_usd_per_bushel": _distribution_summary(
            results.scenarios["harvest_futures_usd_per_bushel"]
        ),
        "basis_distribution_usd_per_bushel": _distribution_summary(
            results.scenarios["basis_usd_per_bushel"]
        ),
        "cash_price_distribution_usd_per_bushel": _distribution_summary(
            results.scenarios["cash_price_usd_per_bushel"]
        ),
    }
    paths["diagnostics"].write_text(
        json.dumps(_json_ready(distribution_diagnostics), indent=2),
        encoding="utf-8",
    )

    if save_trial_level_results:
        paths["scenarios"] = tables / "common_scenarios_10000.csv.gz"
        results.scenarios.to_csv(paths["scenarios"], index=False, compression="gzip")
        paths["main_strategy_results"] = (
            tables / "main_strategy_results_70000.csv.gz"
        )
        results.main_strategy_results.to_csv(
            paths["main_strategy_results"], index=False, compression="gzip"
        )
        paths["robustness_strategy_results"] = (
            tables / "july_beta_zero_strategy_results_70000.csv.gz"
        )
        results.robustness_strategy_results.to_csv(
            paths["robustness_strategy_results"],
            index=False,
            compression="gzip",
        )

    if create_plots:
        paths["strategy_figure"] = figures / "strategy_comparison.png"
        plot_strategy_comparison(results, paths["strategy_figure"])
        paths["robustness_figure"] = figures / "july_beta_zero_robustness.png"
        plot_robustness(results, paths["robustness_figure"])
    return paths


# %% [markdown]
# ## Final interpretation
#
# The model is an academic risk-management comparison, not a claim that one
# hedge is universally optimal. The recommendation is conditional on:
#
# - 1996–2025 calibration data and only 30 historical crop years;
# - a parsimonious Iowa yield/CBOT price linkage;
# - secondary-source daily futures closes rather than official CME settlements;
# - independently sampled weather and yield residuals;
# - same-year paired July/harvest price residuals;
# - an independent triangular basis draw;
# - fixed production cost per acre;
# - whole-contract rounding and simplified transaction/margin assumptions; and
# - a two-date margin-liquidity proxy rather than a daily margin path.
#
# Under those assumptions and the pre-specified CVaR rule, the locked benchmark
# selects **Fixed 75%**. Fixed 100% has higher average profit in this particular
# simulation but fails the 10% over-hedging safeguard. The recommendation remains
# Fixed 75% when the weak July price-signal coefficient is set to zero.

# %%
def _build_parser(default_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the consolidated Iowa corn hedging Lessons 1–11 framework."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Project folder containing processed/iowa_calibration_panel_1996_2025.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder; defaults to <project-root>/master_outputs.",
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=8_122_026)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip the compressed scenario-level result files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG output figures.",
    )
    return parser


def main() -> None:
    default_root = Path(__file__).resolve().parent
    args = _build_parser(default_root).parse_args()
    config = ModelConfig(n_simulations=args.iterations, random_seed=args.seed)
    results = run_master_framework(args.project_root, config)
    output_dir = args.output_dir or (args.project_root / "master_outputs")
    paths = write_outputs(
        results,
        output_dir,
        save_trial_level_results=not args.summary_only,
        create_plots=not args.no_plots,
    )

    display_columns = [
        "strategy",
        "Expected Profit ($/acre)",
        "Profit Std Dev ($/acre)",
        "P5 Profit ($/acre)",
        "CVaR 5% Profit ($/acre)",
        "Probability Overhedged",
        "Probability Margin Call Proxy",
        "Eligible",
        "Preferred",
    ]
    print("\nMAIN MODEL — STRATEGY COMPARISON")
    print(results.main_summary[display_columns].round(4).to_string(index=False))
    print(f"\nPreferred strategy: {results.main_decision.preferred_strategy}")
    print(
        "July beta = 0 preferred strategy: "
        f"{results.robustness_decision.preferred_strategy}"
    )
    print(
        "Recommendation unchanged: "
        f"{results.diagnostics['recommendation_stable_when_july_beta_is_zero']}"
    )
    print("\nSaved outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


# In the companion notebook, running all cells executes this concise benchmark
# block. In normal Python execution, the command-line entry point below runs.
if "__file__" not in globals():
    notebook_project_root = Path.cwd()
    notebook_results = run_master_framework(notebook_project_root)
    notebook_paths = write_outputs(
        notebook_results,
        notebook_project_root / "master_outputs",
    )
    notebook_columns = [
        "strategy",
        "Expected Profit ($/acre)",
        "Profit Std Dev ($/acre)",
        "P5 Profit ($/acre)",
        "CVaR 5% Profit ($/acre)",
        "Probability Overhedged",
        "Probability Margin Call Proxy",
        "Eligible",
        "Preferred",
    ]
    print(notebook_results.main_summary[notebook_columns].round(4).to_string(index=False))
    print("\nPreferred strategy:", notebook_results.main_decision.preferred_strategy)
    print(
        "July beta = 0 preferred strategy:",
        notebook_results.robustness_decision.preferred_strategy,
    )
    print("Outputs saved to:", notebook_project_root / "master_outputs")


if __name__ == "__main__" and "__file__" in globals():
    main()
