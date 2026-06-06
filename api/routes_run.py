from flask import Blueprint, request, Response, stream_with_context, jsonify
from api.run_manager import run_manager, run_wrapper
import threading
import json
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from genetic_algorithm import GeneticAlgorithm
from signal_generator import SignalGenerator
from execution_optimizer import simulate_execution, ExecutionGenome

run_bp = Blueprint('run', __name__)

@run_bp.route('/api/run', methods=['POST'])
def run_strategy():
    config = request.json
    run_id = run_manager.create_run()
    
    # Start the GA in a background thread
    thread = threading.Thread(target=execute_ga_pipeline, args=(run_id, config))
    thread.daemon = True
    thread.start()
    
    return jsonify({"run_id": run_id}), 202

@run_bp.route('/api/progress/<run_id>', methods=['GET'])
def get_progress(run_id):
    q = run_manager.get_queue(run_id)
    if not q:
        return jsonify({"error": "Run not found"}), 404

    def stream():
        while True:
            try:
                msg = q.get(timeout=30)
                if msg == "EOF":
                    break
                
                if "Gen" in msg and "|" in msg:
                    parts = [p.strip() for p in msg.split("|")]
                    gen = parts[0].replace("Gen", "").strip()
                    valid = parts[1].replace("Valid:", "").strip()
                    best_f = parts[2].replace("Best Fit:", "").strip()
                    yield f"data: {json.dumps({'type': 'progress', 'gen': gen, 'valid': valid, 'best': best_f, 'message': msg})}\n\n"
                elif "FINAL_JSON:" in msg:
                    payload = json.loads(msg.replace('FINAL_JSON:', ''))
                    yield f"data: {json.dumps({'type': 'result', 'payload': payload})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'log', 'message': msg})}\n\n"
            except:
                break
        run_manager.cleanup_run(run_id)

    return Response(stream_with_context(stream()), content_type='text/event-stream')

@run_bp.route('/api/cancel/<run_id>', methods=['POST'])
def cancel_run(run_id):
    if run_manager.stop_run(run_id):
        return jsonify({"status": "cancelled"})
    return jsonify({"error": "Run not found"}), 404

@run_bp.route('/api/save', methods=['POST'])
def save_strategy():
    try:
        strat_data = request.json
        pool_file = "strategy_pool.json"
        
        pool = []
        if os.path.exists(pool_file):
            with open(pool_file, 'r') as f:
                pool = json.load(f)
        
        # Prepare strategy for pool compatibility
        new_entry = {
            "stock": strat_data.get('stock', 'UNKNOWN'),
            "direction": strat_data.get('direction', 'LONG'),
            "score": float(strat_data.get('fitness', 0.0)),
            "is_return": float(strat_data.get('is_return', 0.0)),
            "oos_return": float(strat_data.get('oos_return', 0.0)),
            "sortino": float(strat_data.get('sortino', 0.0)),
            "raw_strategy": {
                "features": list(strat_data['dna'].keys()),
                "weights": list(strat_data['dna'].values()),
                "activation_threshold": float(strat_data.get('activation_threshold', 0.0)),
                "mode": "advanced"
            },
            "execution_genome": strat_data.get('exit_dna', {}),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        pool.append(new_entry)
        
        with open(pool_file, 'w') as f:
            json.dump(pool, f, indent=4)
            
        return jsonify({"status": "success", "message": f"Strategy for {new_entry['stock']} saved to pool."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def execute_ga_pipeline(run_id, config):
    stock = config['stock']
    timeframe = config.get('timeframe', '1d')
    loader = DataLoader()
    
    # Branch loader based on timeframe
    if timeframe == '15m':
        raw_data = loader.load_local_kite_15m(symbol=stock)
    elif stock == "BTC-USD":
        raw_data = loader.fetch_stocks_yfinance(symbol="BTC-USD", timeframe="4h", limit=10000)
    else:
        raw_data = loader.load_local_kite_data(symbol=stock)
        
    fe = FeatureEngineer()
    
    # Load Nifty index with matching timeframe for Macro Sync
    nifty_df = loader.load_nifty_index(timeframe=timeframe)
    data = fe.generate_features(raw_data, index_df=nifty_df)
    
    # ── Dynamic 75/25 Train-Test Split ───────────────────────────────
    split_idx = int(len(data) * 0.75)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    split_date = data.index[split_idx].strftime("%Y-%m-%d %H:%M:%S")
    print(f">> Dynamic Split: Train {len(train_df)} bars | Test {len(test_df)} bars")
    print(f">> Split Point: {split_date}")
    
    # ── Train Decision Tree Base Signal ──────────────────────────────
    all_allowed = config['features']['locked'] + config['features']['allowed_pool']
    feature_cols = [f for f in all_allowed if f in train_df.columns]
    
    if feature_cols:
        X_train = train_df[feature_cols].iloc[:-1]
        y_up = (train_df['close'].shift(-1) > train_df['close']).astype(int).iloc[:-1]
        y_down = (train_df['close'].shift(-1) < train_df['close']).astype(int).iloc[:-1]
        
        # 1. LONG SENSOR (Steady trend following)
        tree_up = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=80, random_state=42, n_jobs=-1, class_weight='balanced')
        print(f">> Training RF_LONG on {len(X_train)} bars...")
        tree_up.fit(X_train, y_up)
        
        # 2. SHORT SENSOR (Normalizing depth to prevent over-optimization)
        tree_down = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=80, random_state=42, n_jobs=-1, class_weight='balanced')
        print(f">> Training RF_SHORT on {len(X_train)} bars...")
        tree_down.fit(X_train, y_down)
        
        # Apply to full dataset
        data['tree_signal_up'] = tree_up.predict_proba(data[feature_cols])[:, 1]
        data['tree_signal_down'] = tree_down.predict_proba(data[feature_cols])[:, 1]
        print(">> Dual-Key ML training complete.")
    else:
        # Fallback if no features
        data['tree_signal_up'] = 0.5
        data['tree_signal_down'] = 0.5
        
    # Re-split to ensure train_df has the new column
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    # ─────────────────────────────────────────────────────────────────
    
    def _format_dna(strat):
        dna = {}
        dir_char = '>' if strat.get('direction', 'LONG') == 'LONG' else '<'
        dna[f"ML_PROB_UP {dir_char}"] = round(strat.get('tree_threshold', 0.5), 3)
        for r in strat.get('rules', []):
            dna[f"{r[0]} {r[1]}"] = round(r[2], 3)
        return dna
        
    def _build_result(strat, idx, signals, effective_genome, data, split_date, exit_conf):
        """Helper: evaluate one strategy on the full dataset and return a result dict."""
        trades_raw, final_cap, _ = simulate_execution(signals, effective_genome, data, capital=10000.0)

        mapped_trades = []
        current_eq   = 10000.0
        equity_curve = [{"date": data.index[0].strftime("%Y-%m-%d"), "equity": 10000.0}]
        trade_exits  = {}

        time_format = "%Y-%m-%d %H:%M" if timeframe == '15m' else "%Y-%m-%d"
        for t in trades_raw:
            entry_date = data.index[t['entry_idx']].strftime(time_format)
            exit_date  = data.index[t['exit_idx']].strftime(time_format)
            trade_exits[exit_date] = trade_exits.get(exit_date, 0.0) + t['pnl']
            mapped_trades.append({
                "entry_date": entry_date, "exit_date": exit_date,
                "direction": t.get('direction', 'LONG'),
                "entry_price": t.get('entry_price', 0.0), "exit_price": t.get('exit_price', 0.0),
                "sl_price": t.get('sl_price', 0.0),
                "pnl": round(t['pnl'], 2), "pnl_pct": t.get('pnl_pct', 0.0),
                "reason": t['reason'], "duration": int(t['duration']),
                "trust_score": t.get('trust_score', 0.5),
                "portfolio_impact_pct": t.get('portfolio_impact_pct', 0.0)
            })

        dates = data.index.strftime(time_format).tolist()
        for d in dates:
            if d in trade_exits:
                current_eq += trade_exits[d]
            equity_curve.append({"date": d, "equity": round(current_eq, 2)})

        is_trades  = [t for t in mapped_trades if t['exit_date'] <  split_date]
        oos_trades = [t for t in mapped_trades if t['exit_date'] >= split_date]

        g = effective_genome
        exit_dna = {
            "sl_basis": g.sl_basis, "sl_mult": round(g.sl_mult, 2),
            "tp_mult": round(g.exit_mult, 2),
            "type": "AI-Discovered" if exit_conf.get('stop_loss_type') == 'AUTO' else "Forced"
        }

        is_wins      = len([t for t in is_trades  if t['pnl'] > 0])
        oos_wins     = len([t for t in oos_trades if t['pnl'] > 0])
        is_win_rate  = round((is_wins  / len(is_trades))  * 100, 1) if is_trades  else 0.0
        oos_win_rate = round((oos_wins / len(oos_trades)) * 100, 1) if oos_trades else 0.0

        return {
            "rank": idx + 1,
            "fitness":      round(strat['fitness'], 4),
            "sortino":       round(strat['metrics']['sortino'], 2),
            "direction":    strat.get('direction', 'LONG'),
            "total_return": round(((current_eq - 10000.0) / 10000.0) * 100, 1),
            "is_return":    round((sum([t['pnl'] for t in is_trades])  / 10000.0) * 100, 1) if is_trades  else 0.0,
            "oos_return":   round((sum([t['pnl'] for t in oos_trades]) / 10000.0) * 100, 1) if oos_trades else 0.0,
            "is_win_rate":  is_win_rate, "oos_win_rate": oos_win_rate,
            "max_drawdown": round(strat['metrics']['max_drawdown'] * 100, 1),
            "trades_count": len(mapped_trades),
            "profit_factor": round(strat['metrics']['profit_factor'], 2),
            "dna":          _format_dna(strat),
            "activation_threshold": round(strat.get('tree_threshold', 0.5), 3),
            "exit_dna": exit_dna, "equity_curve": equity_curve, "trade_log": mapped_trades[-100:]
        }

    def inner_work(stop_event=None):
        ga_params  = config.get('ga_params', {})
        exit_conf  = config.get('exit', {})
        all_allowed = config['features']['locked'] + config['features']['allowed_pool']

        # ── Base genome from UI sliders ───────────────────────────────────
        genome = ExecutionGenome(
            sl_basis=exit_conf.get('stop_loss_type', 'ATR'),
            sl_mult=float(exit_conf.get('sl_multiplier', 2.0)),
            exit_type='Fixed_RR',
            exit_mult=float(exit_conf.get('take_profit_multiplier', 2.5))
        )

        # ── PASS 1: Regular mixed-direction GA (top 10) ───────────────────
        print("PASS 1 | Mixed-Direction Discovery")
        ga = GeneticAlgorithm(
            stock_df=train_df, symbol=stock,
            pop_size=ga_params.get('population_size', 120),
            generations=ga_params.get('generations', 15),
            mutation_rate=0.35,
            timeframe=timeframe
        )
        ga.generator.features = {f: (data[f].min(), data[f].max()) for f in all_allowed if f in data.columns}
        population = ga.run_evolution()
        if not population:
            return

        # ── Evaluate results on full dataset ─────────────────────────────
        sg = SignalGenerator(data)
        all_results = []

        def apply_genome(strat, base_genome):
            # If the strategy has an AI-evolved genome, use it fully without overwriting it!
            if 'execution_genome' in strat:
                eg = strat['execution_genome']
            else:
                import copy
                eg = copy.deepcopy(base_genome)
                
            eg.direction = strat.get('direction', 'LONG')
            return eg

        # Take Top 15 candidates (already interleaved Long/Short by the GA)
        for i, strat in enumerate(population[:15]):
            if stop_event and stop_event.is_set():
                return
            signals = sg.generate_signal_vector(strat)
            eg = apply_genome(strat, genome)
            res = _build_result(strat, i, signals, eg, data, split_date, exit_conf)
            all_results.append(res)
            
        print(f"Discovery complete. Sent {len(all_results)} balanced strategies to UI.")
        print(f"FINAL_JSON:{json.dumps({'status': 'success', 'results': all_results, 'split_date': split_date})}")

    run_wrapper(run_id, run_manager, inner_work)
