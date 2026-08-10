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

## How to Run the Master Framework

### Requirements

Use Python 3.10 or later. The framework requires:

- NumPy
- pandas
- Matplotlib
- JupyterLab or Jupyter Notebook if using the notebook interface

Keep the following files together:

```text
project_folder/
├── Iowa_Corn_Hedging_Master_Framework.py
raw/
├── Iowa_Corn_Hedging_Master_Framework.ipynb
├── processed/
│   └── iowa_calibration_panel_1996_2025.csv
└── master_outputs/                        # Created automatically
```

The master run uses the processed calibration panel. The files in `raw/` are retained for provenance but are not required during a normal simulation run.

### 1. Create a Python environment

From the project folder, create and activate a virtual environment.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib jupyterlab
```

### 2. Run the Jupyter notebook

Start JupyterLab from the project folder:

```bash
jupyter lab
```

Open:

```text
Iowa_Corn_Hedging_Master_Framework.ipynb
```

Then select:

```text
Kernel → Restart Kernel and Run All Cells
```

The notebook contains one executable runner cell. It loads the complete implementation from `Iowa_Corn_Hedging_Master_Framework.py`, runs the 10,000-scenario benchmark, validates the results, and displays the strategy comparison.

Do not move the notebook away from the master Python file and `processed/` folder unless their paths are updated.

### 3. Run the framework directly from Python

The same analysis can be run without Jupyter:

```bash
python Iowa_Corn_Hedging_Master_Framework.py
```

The default run uses:

- 10,000 Monte Carlo scenarios;
- random seed `8122026`;
- all seven hedge strategies; and
- the July-price-coefficient robustness test.

A successful run should report:

```text
Preferred strategy: Fixed 75%
July beta = 0 preferred strategy: Fixed 75%
Recommendation unchanged: True
```

### Optional command-line settings

Run a smaller development simulation:

```bash
python Iowa_Corn_Hedging_Master_Framework.py \
  --iterations 1000 \
  --summary-only \
  --no-plots
```

Use a different random seed:

```bash
python Iowa_Corn_Hedging_Master_Framework.py --seed 12345
```

Write results to a different folder:

```bash
python Iowa_Corn_Hedging_Master_Framework.py \
  --output-dir alternative_outputs
```

Available options can be displayed with:

```bash
python Iowa_Corn_Hedging_Master_Framework.py --help
```

### 4. Review the generated outputs

The standard run creates:

```text
master_outputs/
├── figures/
│   ├── strategy_comparison.png
│   └── july_beta_zero_robustness.png
└── tables/
    ├── calibration_diagnostics.csv
    ├── historical_shock_library.csv
    ├── model_parameters.json
    ├── framework_diagnostics.json
    ├── common_scenarios_10000.csv.gz
    ├── main_strategy_results_70000.csv.gz
    ├── main_strategy_summary.csv
    ├── july_beta_zero_strategy_results_70000.csv.gz
    ├── july_beta_zero_strategy_summary.csv
    └── final_decision.json
```

The most important files are:

- `main_strategy_summary.csv` for the seven-strategy comparison;
- `final_decision.json` for the recommendation and decision rules;
- `framework_diagnostics.json` for model checks and distribution summaries; and
- `strategy_comparison.png` for the main visual results.

### 5. Run the automated tests

If the development dependencies are installed, run:

```bash
python -m pytest
```

The tests verify reproducibility, common random numbers, futures P&L accounting, strategy rules, benchmark values, and the July-beta-zero robustness result.

### Troubleshooting

#### `FileNotFoundError` for the project folder or calibration panel

Confirm that these items are in the same project folder:

```text
Iowa_Corn_Hedging_Master_Framework.py
Iowa_Corn_Hedging_Master_Framework.ipynb
processed/iowa_calibration_panel_1996_2025.csv
```

Start Jupyter from that folder rather than from a parent directory or Downloads.

#### `ModuleNotFoundError`

Install the required packages in the active environment:

```bash
python -m pip install numpy pandas matplotlib jupyterlab
```

Ensure Jupyter is using the same Python environment in which the packages were installed.

#### A notebook cell reports that a name is undefined

Close any older copy of the notebook, reopen the current master notebook, and select:

```text
Kernel → Restart Kernel and Run All Cells
```

The current notebook contains only one executable runner cell and should not produce cell-order errors.

#### Results differ from the benchmark

Confirm that the default values have not been changed:

```text
Iterations: 10,000
Random seed: 8,122,026
March futures price: $4.70/bu
Farm acres: 1,000
Production cost: $911.98/acre
```

Changing the seed, simulation count, costs, or model assumptions can change the reported strategy metrics.

## Project status

Data collection and calibration inputs are present. The simulation engine, strategy comparison, robustness checks, and final results remain to be implemented.

## Disclaimer

This repository is an academic project. Its results should not be interpreted as individualized trading, hedging, or financial advice.
