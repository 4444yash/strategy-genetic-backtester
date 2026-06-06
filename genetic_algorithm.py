import random
import copy
import pandas as pd
import numpy as np
from strategy_generator import StrategyGenerator
from execution_optimizer import simulate_execution
from signal_generator import SignalGenerator
import os

class GeneticAlgorithm:
    def __init__(self, stock_df: pd.DataFrame, symbol: str, regime_name: str = "Trend",
                 pop_size=100, generations=25, mutation_rate=0.3, timeframe='1d'):
        self.stock_df = stock_df.copy()
        self.symbol = symbol
        self.regime_name = regime_name
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.timeframe = timeframe
        self.generator = StrategyGenerator()
        
        # ── PART 2: Regime-Aware Splitting ──────────────────────────────
        # Segment data by the specific regime
        if 'Active_Regime' in self.stock_df.columns:
            # Shifted already in the loader, but safety check
            self.regime_df = self.stock_df[self.stock_df['Active_Regime'] == self.regime_name].copy()
        else:
            self.regime_df = self.stock_df
            
        self.num_segments = 3
        segment_size = len(self.regime_df) // self.num_segments
        if segment_size > 0:
            self.segment_dfs = [self.regime_df.iloc[i * segment_size : (i+1) * segment_size] for i in range(self.num_segments)]
            self.signal_generators = [SignalGenerator(seg_df, timeframe=self.timeframe) for seg_df in self.segment_dfs]
        else:
            self.segment_dfs = []
            self.signal_generators = []

        # Dynamically build feature ranges
        raw_price_features = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp']
        for col in self.regime_df.select_dtypes(include=[np.number]).columns:
            if col.lower() not in raw_price_features and col not in self.generator.features:
                self.generator.features[col] = (float(self.regime_df[col].min()), float(self.regime_df[col].max()))

    def fitness_function(self, metrics: dict, strategy: dict) -> float:
        """
        Anti-Overfitting Fitness Function - Smooth Penalty Edition.
        """
        total_trades = metrics.get('number_of_trades', 0)
        max_dd = metrics.get('max_drawdown', 1.0)
        penalties = {}
        
        # ── Evolutionary Gravity: Hard Kill ──────────────────
        if total_trades == 0 or max_dd > 0.60:
            strategy['penalties'] = {'hard_kill': 'no_trades_or_max_dd'}
            return -99999.0

        # 1. Hard Filter for Trend (Momentum purity)
        if self.regime_name == "Trend" and metrics.get('expansion_ratio', 1.0) < 1.03:
            strategy['penalties'] = {'hard_kill': 'trend_weak_expansion_strict'}
            return -99999.0

        # Part C: Hard Reject for Squeeze Variance
        if self.regime_name == "Squeeze" and metrics.get('segment_std', 0.0) > 0.12:
            strategy['penalties'] = {'hard_kill': 'squeeze_high_variance_strict'}
            return -99999.0

        # ── Metrics (Part 1 & 8) ──────────────────────────────
        sortino = min(metrics.get('sortino', 0.0), 5.0)
        raw_pf = metrics.get('profit_factor', 1.0)
        pf = min(raw_pf, 3.5)
        net_return = min(metrics.get('net_return', 0.0), 0.3)
        
        # Base Fitness
        fitness = (1.0 * sortino) + (0.5 * pf) + (1.0 * net_return) - (0.2 * max_dd)

        # ── Overtrading Penalty ───────────────────────────────
        if self.regime_name == "Squeeze":
            fitness -= 0.01 * total_trades
        elif self.regime_name == "MeanRev":
            fitness -= 0.005 * total_trades
        elif self.regime_name == "Trend":
            fitness -= 0.003 * total_trades

        # ── Modifiers ─────────────────────────────────────────
        # Trade Factor
        trade_factor = min(total_trades / 40.0, 1.0)
        fitness *= trade_factor
        penalties['trade_factor'] = trade_factor
        
        if total_trades < 15:
            fitness *= 0.3
            penalties['low_trades_global'] = 0.3
            
        if self.regime_name == "Squeeze" and total_trades < 25:
            fitness *= 0.2
            penalties['low_trades_squeeze'] = 0.2
        elif self.regime_name == "MeanRev" and total_trades < 30:
            fitness *= 0.3
            penalties['low_trades_meanrev'] = 0.3
        elif self.regime_name == "Trend" and total_trades < 25:
            fitness *= 0.5
            penalties['low_trades_trend'] = 0.5

        # ── Profit Factor Stabilization (Part 1) ──────────────
        if raw_pf > 4.5:
            strategy['penalties'] = {'hard_kill': 'unrealistic_pf_spike'}
            return -99999.0
            
        if raw_pf < 1.25:
            fitness *= 0.4
            penalties['garbage_pf'] = 0.4

        # ── Global Quality Filters (Part 1) ───────────────────
        avg_trade_return = net_return / max(total_trades, 1)
        if avg_trade_return < 0.002:
            fitness *= 0.7
            penalties['low_edge_strength'] = 0.7
            
        if net_return > 0.20:
            fitness *= 1.1
            penalties['strong_return_bonus'] = 1.1

        # ── Segment Balance (Part 5) ──────────────────────────
        segment_returns = metrics.get('segment_returns', [])
        profitable_segments = sum(1 for r in segment_returns if r > 0)
        
        if profitable_segments < 2:
            fitness *= 0.5
            penalties['weak_segments'] = 0.5
        if any(r < -0.02 for r in segment_returns):
            fitness *= 0.7
            penalties['catastrophic_segment_dd'] = 0.7
        if len(segment_returns) > 0 and all(r > 0 for r in segment_returns):
            fitness *= 1.1
            penalties['segment_consistency_bonus'] = 1.1

        if len(segment_returns) > 0:
            max_seg = max(segment_returns)
            min_seg = min(segment_returns)
            if abs(max_seg) > 4 * (abs(min_seg) + 1e-6):
                fitness *= 0.6
                penalties['segment_dominance'] = 0.6
            
            # --- STABILITY PENALTY ---
            seg_std = metrics.get('segment_std', 0.0)
            if seg_std > 0.08:
                stability_penalty = 1.0 - (seg_std * 2.0)
                fitness *= max(0.5, stability_penalty)
                penalties['stability_guard'] = round(max(0.5, stability_penalty), 2)
            
            if metrics.get('segment_std', 0.0) > 0.12:
                fitness *= 0.8
                penalties['high_segment_variance'] = 0.8

        # ── Rule Quality (Part 7) ─────────────────────────────
        num_rules = metrics.get('rules_count', len(strategy.get('rules', [])))
        if num_rules < 2:
            fitness *= 0.85
            penalties['trivial_rules'] = 0.85

        # ── Absolute Feature Ban ──────────────────────────────
        raw_price_features = ['open', 'high', 'low', 'close', 'volume']
        using_raw = any(r[0].lower() in raw_price_features for r in strategy.get('rules', []))
        if using_raw:
            fitness *= 0.5
            penalties['raw_absolute_feature'] = 0.5

        # ── Wrong Genome & Soft SMA Bias ──────────────────────
        bias_map = {"Trend": ["MACD", "MA", "ADX"], "MeanRev": ["RSI", "BB", "stoch"], "Squeeze": ["BB_width", "ATR"]}
        preferred = bias_map.get(self.regime_name, [])
        used_features = [r[0].upper() for r in strategy.get('rules', [])]
        has_preferred = any(p in f for p in preferred for f in used_features)
        if not has_preferred and preferred:
            fitness *= 0.90
            penalties['wrong_genome'] = 0.90

        if self.regime_name in ["Trend", "Squeeze"] and 'price_minus_MA200' in self.regime_df.columns:
            avg_dist = self.regime_df['price_minus_MA200'].mean()
            weight = 0.2 if self.regime_name == "Trend" else 0.05
            fitness += weight * avg_dist

        # ── Regime Filters (Part 2, 3, 4) ─────────────────────
        exp_ratio = metrics.get('expansion_ratio', 1.0)
        
        if self.regime_name == "Trend":
            if exp_ratio > 1.08:
                fitness *= 1.05
                penalties['trend_strong_expansion_bonus'] = 1.05
                
            if net_return < 0.08:
                fitness *= 0.6
                penalties['trend_weak_momentum'] = 0.6
                
        elif self.regime_name == "MeanRev":
            if exp_ratio > 1.08:
                fitness *= 0.7
                penalties['meanrev_vol_expansion'] = 0.7
            if metrics.get('segment_std', 0.0) > 0.12:
                fitness *= 0.7
                penalties['meanrev_unstable'] = 0.7
                
        elif self.regime_name == "Squeeze":
            if exp_ratio < 1.05:
                fitness *= 0.6
                penalties['squeeze_weak_expansion'] = 0.6
            if net_return < 0.03:
                fitness *= 0.5
                penalties['squeeze_low_return'] = 0.5
                
            # Squeeze Specific Checks
            avg_bars_held = metrics.get('avg_bars_held', 0)
            if avg_bars_held < 3:
                fitness *= 0.6
                penalties['squeeze_too_fast'] = 0.6
                
            regime_density = total_trades / len(self.regime_df) if len(self.regime_df) > 0 else 0
            if regime_density > 0.1:
                fitness *= 0.5
                penalties['squeeze_overtrading'] = 0.5
                
            if regime_density < 0.08:
                patience_multiplier = 1.0
            elif metrics.get('avg_wait', 0) > 10:
                patience_multiplier = 1.1
            else:
                patience_multiplier = 1.0
            fitness *= patience_multiplier
            if patience_multiplier > 1.0:
                penalties['squeeze_patience_bonus'] = patience_multiplier

        # Net Profitability Enforcement
        if net_return <= 0:
            fitness *= 0.5
            penalties['negative_net_return'] = 0.5

        # --- FINAL TRUTH GATE ---
        if np.isnan(fitness) or fitness is None:
            fitness = -999.0

        strategy['penalties'] = penalties
        return float(fitness)

    def _evaluate_strategy(self, strategy: dict) -> dict:
        if not self.segment_dfs:
            strategy['fitness'] = -999.0
            return strategy

        total_trades = 0
        pfs, dds, segment_sortinos, segment_returns = [], [], [], []
        all_trades = []
        
        from execution_optimizer import ExecutionGenome
        genome = strategy.get('execution_genome')
        if not genome:
            genome = ExecutionGenome.random_genome()
            strategy['execution_genome'] = genome
        genome.direction = strategy.get('direction', 'LONG')

        for seg_df, sg in zip(self.segment_dfs, self.signal_generators):
            sig_vector = sg.generate_signal_vector(strategy)
            pos_size_arr = seg_df['Position_Size'].values if 'Position_Size' in seg_df.columns else None
            conf_arr = seg_df['Confidence'].values if 'Confidence' in seg_df.columns else None
            
            trades, final_cap, seg_max_dd = simulate_execution(
                sig_vector, genome, seg_df, timeframe=self.timeframe, 
                pos_size_multiplier=pos_size_arr,
                confidence_series=conf_arr
            )
            
            num_trades = len(trades)
            total_trades += num_trades
            
            if num_trades > 0:
                pnls = np.array([t['pnl'] for t in trades])
                seg_ret = pnls.sum() / 10000.0
                segment_returns.append(seg_ret)
                
                pct_returns = np.array([t['pnl'] / (t['entry_price'] * t['position_size']) for t in trades if t['entry_price'] > 0 and t['position_size'] > 0])
                segment_sortinos.append(self._calc_sortino(pct_returns))
                
                wins, losses = pnls[pnls > 0], np.abs(pnls[pnls < 0])
                pfs.append(wins.sum() / losses.sum() if len(losses) > 0 else 2.0)
                
                # Ensure seg_max_dd is valid float
                valid_dd = float(seg_max_dd) if seg_max_dd is not None else 0.0
                dds.append(valid_dd)
                
                # Squeeze metrics tracking
                atr_arr = seg_df['ATR'].values if 'ATR' in seg_df.columns else np.zeros(len(seg_df))
                last_exit = 0
                for t in trades:
                    entry, exit_idx = t['entry_idx'], t['exit_idx']
                    wait = entry - last_exit if last_exit > 0 else 0
                    last_exit = exit_idx
                    
                    pre_vol = atr_arr[entry] if entry < len(atr_arr) else 0.01
                    post_idx = min(entry + 10, len(atr_arr))
                    post_vol = np.mean(atr_arr[entry:post_idx]) if post_idx > entry else pre_vol
                    
                    all_trades.append({
                        'duration': t['duration'],
                        'wait': wait,
                        'pre_vol': pre_vol,
                        'post_vol': post_vol
                    })
            else:
                segment_returns.append(0.0)
                segment_sortinos.append(0.0)
                pfs.append(1.0)
                dds.append(0.0)

        metrics = {
            'number_of_trades': total_trades,
            'max_drawdown': np.max(dds) if dds else 0.0,
            'sortino': np.mean(segment_sortinos) if segment_sortinos else 0.0,
            'profit_factor': np.mean(pfs) if pfs else 1.0,
            'net_return': sum(segment_returns),
            'segment_returns': segment_returns,
            'segment_std': np.std(segment_returns) if segment_returns else 0.0,
            'rules_count': len(strategy.get('rules', [])),
            'avg_bars_held': np.mean([t['duration'] for t in all_trades]) if all_trades else 0.0,
            'avg_wait': np.mean([t['wait'] for t in all_trades if t['wait'] > 0]) if all_trades else 0.0
        }
        
        avg_pre = np.mean([t['pre_vol'] for t in all_trades]) if all_trades else 0.01
        avg_post = np.mean([t['post_vol'] for t in all_trades]) if all_trades else 0.01
        metrics['expansion_ratio'] = float(avg_post / avg_pre) if avg_pre > 0 else 1.0

        strategy['metrics'] = metrics
        strategy['fitness'] = self.fitness_function(metrics, strategy)
        return strategy

    def _calc_sortino(self, pnls: np.ndarray) -> float:
        if len(pnls) < 2: return 0.0
        downside_diff = np.minimum(pnls, 0.0)
        downside_dev = np.sqrt(np.mean(downside_diff ** 2))
        if downside_dev == 0: return 2.0
        return (pnls.mean() / downside_dev) * np.sqrt(252)

    def run_evolution(self):
        # Initial Population
        self.population = []
        for _ in range(self.pop_size):
            strat = self.generator.generate_random_strategy(timeframe=self.timeframe)
            from execution_optimizer import ExecutionGenome
            strat['execution_genome'] = ExecutionGenome.random_genome()
            self.population.append(strat)

        for gen in range(self.generations):
            # Evaluate
            for i in range(len(self.population)):
                self.population[i] = self._evaluate_strategy(self.population[i])
            
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Reporting
            best = self.population[0]
            pen_str = ", ".join([f"{k}: {v}" for k, v in best.get('penalties', {}).items()])
            print(f"Gen {gen} | Regime: {self.regime_name} | Best Fit: {best['fitness']:.3f} | Trades: {best['metrics']['number_of_trades']} | Pen: {pen_str}")

            # Evolve
            elites = self.population[:int(self.pop_size * 0.2)]
            next_gen = elites.copy()
            
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_gen.append(child)
            
            self.population = next_gen
        return self.population[:10] # Top 10

    def _init_population(self):
        return [self.generator.generate_random_strategy(regime_name=self.regime_name, timeframe=self.timeframe) for _ in range(self.pop_size)]

    def _crossover(self, p1, p2):
        child = copy.deepcopy(p1)
        if random.random() > 0.5:
            child['rules'] = copy.deepcopy(p2['rules'])
        return child

    def _mutate(self, strat):
        if random.random() < self.mutation_rate:
            strat['rules'] = self.generator.generate_random_strategy(regime_name=self.regime_name, timeframe=self.timeframe)['rules']
        return strat

def load_and_prep_data(ticker: str):
    """ Part 2.1 & 2.2: Merge Allocator Data & Shift """
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    
    loader = DataLoader()
    raw_df = loader.fetch_stocks_yfinance(symbol=f"{ticker}.NS", limit=3000)
    fe = FeatureEngineer()
    df = fe.generate_features(raw_df)
    
    allocator_file = 'master_allocation_engine.csv'
    if os.path.exists(allocator_file):
        alloc_df = pd.read_csv(allocator_file)
        alloc_df['Date'] = pd.to_datetime(alloc_df['Date']).dt.date
        
        # Filter for this ticker
        alloc_df = alloc_df[alloc_df['Ticker'] == ticker]
        
        # Merge
        df.reset_index(inplace=True)
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'Date'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = pd.merge(df, alloc_df[['Date', 'Active_Regime', 'Position_Size']], on='Date', how='inner')
        
        # ── Part 2.2: Prevent Lookahead Bias (CRITICAL) ─────────────
        df['Active_Regime'] = df['Active_Regime'].shift(1)
        df['Position_Size'] = df['Position_Size'].shift(1)
        df.dropna(subset=['Active_Regime', 'Position_Size'], inplace=True)
        df.set_index('Date', inplace=True)
        print(f"Data aligned. Total overlapping days: {len(df)}")
    else:
        # Fallback if no allocator data: dummy regime
        df['Active_Regime'] = 'Trend'
        df['Position_Size'] = 1.0
        
    return df

def strategy_similarity(a, b):
    rules_a = set([r[0] for r in a.get('rules', [])])
    rules_b = set([r[0] for r in b.get('rules', [])])
    overlap_ratio = len(rules_a.intersection(rules_b)) / max(len(rules_a), len(rules_b)) if rules_a and rules_b else 0.0
    
    g_a = a.get('execution_genome')
    g_b = b.get('execution_genome')
    exec_sim = 0.0
    if g_a and g_b:
        sim_points = 0
        if getattr(g_a, 'sl_basis', '') == getattr(g_b, 'sl_basis', ''): sim_points += 1
        if abs(getattr(g_a, 'sl_mult', 0) - getattr(g_b, 'sl_mult', 0)) < 0.2: sim_points += 1
        if getattr(g_a, 'exit_type', '') == getattr(g_b, 'exit_type', ''): sim_points += 1
        if abs(getattr(g_a, 'exit_mult', 0) - getattr(g_b, 'exit_mult', 0)) < 0.2: sim_points += 1
        if abs(getattr(g_a, 'tp_mult', 0) - getattr(g_b, 'tp_mult', 0)) < 0.2: sim_points += 1
        exec_sim = sim_points / 5.0
        
    return (overlap_ratio * 0.7) + (exec_sim * 0.3)

def run_multi_agent_system(ticker: str):
    df = load_and_prep_data(ticker)
    
    strategy_bank = {}
    
    for regime in ['Trend', 'MeanRev', 'Squeeze']:
        print(f"\nStarting GA for {ticker} - Regime: {regime}")
        ga = GeneticAlgorithm(df, ticker, regime_name=regime, pop_size=100, generations=25)
        
        # Minimum Data Requirement (Part 2.5)
        if len(ga.regime_df) < 20:
            print(f"Skipping {regime}: Insufficient data.")
            continue
            
        top_models = ga.run_evolution()
        
        # ── Diversity Enforcement (Top 3 FIX) ──────────────────
        selected_strategies = []
        for candidate in top_models:
            # Check similarity against already selected elites
            if not any(strategy_similarity(candidate, s) > 0.8 for s in selected_strategies):
                selected_strategies.append(candidate)
            if len(selected_strategies) == 3:
                break
                
        strategy_bank[regime] = selected_strategies
        
    return strategy_bank

def ensemble_signal(date, ticker, current_regime, strategy_bank, df):
    """ Part 6.2: Live Signal Function """
    models = strategy_bank.get(current_regime, [])
    if not models:
        return "NO TRADE"
    
    # Simple majority vote (2 out of 3)
    votes = 0
    sg = SignalGenerator(df) # Logic for current bar
    
    for model in models:
        # In a real live scenario, we evaluate the model's rules on current data
        # Here we simulate by looking at the last signal generated
        sig_vector = sg.generate_signal_vector(model)
        if sig_vector[-1]: # Latest bar
            votes += 1
            
    if votes >= 2:
        return "BUY/SELL"
    return "NO TRADE"

def save_strategy_bank(ticker, bank):
    """ Save the strategy bank to a JSON file for persistent use. """
    import json
    
    serializable_bank = {}
    for regime, models in bank.items():
        serializable_models = []
        for model in models:
            m = copy.deepcopy(model)
            if 'execution_genome' in m and hasattr(m['execution_genome'], 'to_dict'):
                m['execution_genome'] = m['execution_genome'].to_dict()
            serializable_models.append(m)
        serializable_bank[regime] = serializable_models

    filename = f'strategy_bank_{ticker}.json'
    with open(filename, 'w') as f:
        json.dump(serializable_bank, f, indent=4)
    print(f"Strategy Bank saved to {filename}")

if __name__ == "__main__":
    ticker = "RELIANCE"
    bank = run_multi_agent_system(ticker)
    save_strategy_bank(ticker, bank)
    
    print("\nEvolution Complete. Strategy Bank Built.")
    for regime, models in bank.items():
        print(f"Regime {regime}: {len(models)} models saved.")
