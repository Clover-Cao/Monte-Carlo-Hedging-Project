# Iowa Corn Hedging Monte Carlo Project

This project studies how Iowa corn producers can use December corn futures to manage the combined effects of weather, crop-yield, price, and basis risk. The final model will compare fixed-percentage hedges with weather-informed dynamic strategies through Monte Carlo simulation.

## Research question

Under uncertainty in growing-season weather, crop yield, and harvest-time prices, what percentage of expected corn production should a producer hedge using corn futures? Can a hedge that is updated when July weather information becomes available reduce downside revenue risk more effectively than a traditional fixed-percentage hedge?

## Team

- Wanzhu Zheng
- Junlin Yu
- Clover (Hongxu) Cao
- Ivy Ding

## Planned analysis

The simulation is designed around a representative 1,000-acre Iowa corn farm and will:

- simulate correlated weather, yield, futures price, and basis outcomes;
- compare fixed hedge ratios of 0%, 25%, 50%, 75%, and 100%;
- compare the fixed strategies with weather-updated and weather-signal strategies;
- use the December corn futures contract and 5,000-bushel contract sizes;
- evaluate 10,000 trials with a fixed random seed for reproducibility; and
- prevent look-ahead bias by using only information available on each hedge date.

The main performance measures are expected revenue, revenue volatility, probability of a loss, 5% Value at Risk (VaR), 5% Conditional Value at Risk (CVaR), over-hedging probability, margin-call risk, and transaction costs.

## Key datasets

| File | Description |
| --- | --- |
| `processed/iowa_calibration_panel_1996_2025.csv` | Combined annual yield, weather, and December futures observations used for calibration |
| `processed/historical_shock_library.csv` | Historical weather, yield-residual, and paired price-residual shocks for bootstrap sampling |
| `processed/calibration_parameters.json` | Selected yield and price models, 2026 baseline, basis distribution, hedge rules, and operating assumptions |
| `processed/calibration_diagnostics.csv` | Model comparison and validation statistics |
| `processed/baseline_2026_preseason_futures.csv` | 2026 preseason December futures price assumption |
| `processed/source_manifest.csv` | Data provenance and source notes |

## Current calibration assumptions

- Calibration period: 1996–2025 (30 crop years)
- Forecast crop year: 2026
- Selected yield model: time trend plus July Palmer Drought Severity Index (PDSI)
- 2026 preseason December futures price: $4.70 per bushel on March 2, 2026
- Basis: triangular distribution from -$0.28 to -$0.11 per bushel, with a -$0.20 mode
- Production cost: $911.98 per acre
- Monte Carlo iterations: 10,000
- Random seed: 8122026

These values are project inputs, not forecasts or financial advice. See `processed/calibration_parameters.json` for the complete machine-readable specification.

## Data sources

- USDA National Agricultural Statistics Service: Iowa corn yield and harvested acreage
- NOAA National Centers for Environmental Information: May–July precipitation, May–July average temperature, and July PDSI
- Barchart public historical interface via TradingCharts: historical December corn futures prices
- CME Group: corn futures contract specifications
- Iowa State University Ag Decision Maker: Iowa harvest basis and 2026 crop-production cost assumptions

Source URLs and notes are recorded in `processed/source_manifest.csv`. Raw files are retained to support traceability.

## Reproducibility guidelines

- Do not edit files in `raw/` manually; preserve them as downloaded.
- Put cleaned or derived datasets in `processed/`.
- Add scripts that build processed data instead of making undocumented spreadsheet edits.
- Use relative paths in code so it works on every teammate's computer.
- Record new sources in `processed/source_manifest.csv`.
- Keep the simulation seed fixed for benchmark results and document any alternative seeds.
- Do not commit credentials, API keys, temporary files, or large generated outputs.

## Project status

Data collection and calibration inputs are present. The simulation engine, strategy comparison, robustness checks, and final results remain to be implemented.

## Disclaimer

This repository is an academic project. Its results should not be interpreted as individualized trading, hedging, or financial advice.
