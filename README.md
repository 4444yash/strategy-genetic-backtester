# Genetic Algorithm Strategy Backtester UI & Engine

An institutional-grade, regime-aware Genetic Algorithm (GA) backtesting suite and strategy discovery engine. This project features a Flask-based backend pipeline and a real-time web UI to evolve, evaluate, filter, and backtest quantitative trading strategies.

---

##  Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your machine.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/4444yash/strategy-genetic-backtester.git
   cd strategy-genetic-backtester
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000` to start evolving strategies.

---

##  Project Structure & File Guide

Here is a breakdown of what each of the core files in the project does:

### Core Application & Frontend
| File | Description |
| :--- | :--- |
| **`app.py`** | Main entrypoint starting the Flask server and registering the API blueprints. |
| **`index.html`** | Single-page UI dashboard. Handles parameter configs, displays evolution logs via SSE (Server-Sent Events), renders interactive charts, and manages the strategy pool. |
| **`requirements.txt`** | Python dependencies (Flask, pandas, numpy, scikit-learn, yfinance, ccxt). |
| **`.gitignore`** | Excludes bulky cached dataset CSVs, logs, databases, and python build artifacts from git tracking. |

### Algorithmic Pipeline
| File | Description |
| :--- | :--- |
| **`data_loader.py`** | Downloads and caches stock (Yahoo Finance) or cryptocurrency (CCXT/Binance) data. Aligns index benchmark calendars dynamically. |
| **`feature_engineer.py`** | Performs feature engineering to generate technical indicators (RSI, MACD, Bollinger Bands, ATR, trend slopes) and macro market regime states. |
| **`strategy_generator.py`** | Defines starting features bounds and generates random rule configurations mapped by regime structures (Trend, Mean Reversion, Squeeze). |
| **`signal_generator.py`** | Vectorizes strategy rules and ML backbone outputs into shift-corrected boolean signal vectors to prevent lookahead bias. |
| **`genetic_algorithm.py`** | Evolves populations of strategies using selection, crossover, and mutation, guided by an anti-overfitting, multi-segment penalty fitness function. |
| **`execution_optimizer.py`** | Event-driven simulation engine testing execution genomes with institutional rules (1% fixed fractional risk, swing stops, profit locks, time limits). |

### API Routes & Utilities
| File | Description |
| :--- | :--- |
| **`api/__init__.py`** | Defines the `api/` directory as a python package. |
| **`api/routes_universe.py`** | Handles universe configuration endpoints (NSE stocks & crypto symbols) and allowed feature lists. |
| **`api/routes_pool.py`** | Manages reading/writing to the strategy pool and downloading benchmark index data. |
| **`api/routes_run.py`** | Manages spawning background threads for running the GA pipeline and streaming progress logs in real-time. |
| **`api/run_manager.py`** | Coordinates background runs and intercepts Python stdout/logs to funnel them into queues for streaming. |
| **`strategy_pool.json`** | Local database storing saved strategy templates containing entry rules and exit genomes. |

---

##  System Features
* **Dual-Key ML Backbone**: Combines RandomForest Classifiers for signal verification with heuristic technical rules.
* **Anti-Overfitting Fitness**: Penalizes overtrading, segment returns inconsistency, and unrealistic spikes in profit factor.
* **Institutional Execution**: Built-in 1% risk rule, trailing stop losses, time-based exits, and regime-based filters.
* **Real-time Logging**: Captures background stdout and sends it directly to the UI using server-sent events.
