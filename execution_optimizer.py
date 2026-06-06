import pandas as pd
import numpy as np

class ExecutionGenome:
    def __init__(self, sl_basis='ATR', sl_mult=2.0, exit_type='trailing_sl', exit_mult=3.0, tp_mult=4.0, direction='LONG'):
        self.sl_basis = sl_basis
        self.sl_mult = sl_mult
        self.exit_type = exit_type
        self.exit_mult = exit_mult
        self.tp_mult = tp_mult
        self.direction = direction

    @staticmethod
    def random_genome():
        import random
        return ExecutionGenome(
            sl_basis=random.choice(['ATR', 'FIXED']),
            sl_mult=random.uniform(1.0, 3.5),
            exit_type=random.choice(['trailing_sl', 'fixed_tp']),
            exit_mult=random.uniform(1.5, 5.0),
            tp_mult=random.uniform(2.0, 6.0),
            direction='LONG'
        )

    def to_dict(self):
        return self.__dict__

def simulate_execution(entry_signals, genome, data, capital=10000, timeframe='1d', 
                      pos_size_multiplier=None, confidence_series=None):
    """
    Institutional V5 Engine - STABILITY TEST VERSION
    - 1% Fixed Risk (No Pyramiding)
    - Trend Max Hold: 70 Days
    - Profit Locking @ 5%
    """
    trades = []
    lows   = data['low'].values
    highs  = data['high'].values
    closes = data['close'].values
    opens  = data['open'].values
    atr    = data['ATR'].values
    
    if 'Active_Regime' in data.columns:
        regimes = data['Active_Regime'].values
    else:
        regimes = np.full(len(data), 'Trend')

    direction = getattr(genome, 'direction', 'LONG')
    entry_indices = np.where(entry_signals)[0]
    last_exit_idx = -1
    
    MAX_HOLD = {"Trend": 70, "MeanRev": 10, "Squeeze": 15}

    for i in entry_indices:
        if i <= last_exit_idx:
            continue
            
        entry_idx = i
        entry_regime = str(regimes[entry_idx])
        
        # 1. 1% Risk Rule
        risk_amount = capital * 0.01
        
        # 2. Price and Stop-Loss
        raw_entry_price = opens[entry_idx]
        slippage = 0.0015
        entry_price = raw_entry_price * (1.0 + slippage) if direction == 'LONG' else raw_entry_price * (1.0 - slippage)
        
        if genome.sl_basis == 'ATR':
            sl_dist = atr[i] * genome.sl_mult
        else:
            sl_dist = entry_price * 0.02
        
        sl_dist = max(sl_dist, entry_price * 0.005) 
        sl_price = entry_price - sl_dist if direction == 'LONG' else entry_price + sl_dist
        
        # 3. Position size
        position_size = int(risk_amount / sl_dist)
        if (position_size * entry_price) > capital:
            position_size = int(capital / entry_price)

        if position_size <= 0: continue
        
        # 4. Simulation
        highest_reached = entry_price
        lowest_reached = entry_price
        exit_idx, exit_reason, exit_price = None, "END_OF_DATA", closes[-1]
        
        for j in range(entry_idx, len(data)):
            curr_low, curr_high, curr_close = lows[j], highs[j], closes[j]
            bars_held = j - entry_idx
            highest_reached = max(highest_reached, curr_high)
            lowest_reached = min(lowest_reached, curr_low)
            
            pnl_pct = 0.0
            if entry_price > 0:
                if direction == 'LONG':
                    pnl_pct = (curr_close - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - curr_close) / entry_price
            
            if np.isnan(pnl_pct): pnl_pct = 0.0
            
            # NO PYRAMIDING HERE (Per User Request)

            # --- PROFIT LOCK (V5 Logic) ---
            if pnl_pct > 0.05:
                lock_price = entry_price * (1 + pnl_pct * 0.5)
                if direction == 'LONG':
                    sl_price = max(sl_price, lock_price)
                else:
                    sl_price = min(sl_price, lock_price)

            # --- EXIT CONDITIONS ---
            if sl_price is None or np.isnan(sl_price):
                sl_price = entry_price * 0.95 if direction == 'LONG' else entry_price * 1.05
                
            if (direction == 'LONG' and curr_low <= sl_price) or (direction == 'SHORT' and curr_high >= sl_price):
                exit_idx, exit_reason, exit_price = j, "SL", sl_price
                break
            
            if pnl_pct < -0.10:
                exit_idx, exit_reason, exit_price = j, "STOP_LOSS_HARD", (entry_price * 0.9 if direction == 'LONG' else entry_price * 1.1)
                break
                
            limit = MAX_HOLD.get(str(entry_regime), 40)
            if bars_held >= limit:
                exit_idx, exit_reason, exit_price = j, "TIME_EXIT", curr_close
                break
            
            if str(regimes[j]) != str(entry_regime):
                exit_idx, exit_reason, exit_price = j, "REGIME_EXIT", curr_close
                break

        # Finalize trade
        if exit_idx is not None:
            trade_return = (exit_price - entry_price) / entry_price if direction == 'LONG' else (entry_price - exit_price) / entry_price
            trade_pnl = position_size * (exit_price - entry_price) if direction == 'LONG' else position_size * (entry_price - exit_price)
            
            trades.append({
                'pnl': trade_pnl,
                'pnl_pct': round(trade_return * 100, 2),
                'duration': exit_idx - entry_idx,
                'reason': exit_reason,
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size
            })
            last_exit_idx = exit_idx
            capital += trade_pnl

    # Final Reporting (Max DD)
    total_max_dd = 0.0
    if trades:
        peak = 10000.0
        running = 10000.0
        for t in trades:
            running += t['pnl']
            peak = max(peak, running)
            dd = (peak - running) / peak
            total_max_dd = max(total_max_dd, dd)

    return trades, capital, total_max_dd
