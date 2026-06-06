import pandas as pd
import numpy as np

class SignalGenerator:
    """
    Separates the logic of evaluating feature comparisons into boolean signal arrays 
    from the execution loop of trade tracking.
    """
    def __init__(self, df: pd.DataFrame, timeframe: str = '1d'):
        self.df = df
        self.df_len = len(df)
        self.timeframe = timeframe
        
        # Cache specific columns used by generators as numpy arrays for max speed
        self._columns_cache = {}
        for col in self.df.columns:
            # We cache everything safely, assuming features exist
            self._columns_cache[col] = self.df[col].to_numpy()
            
    def get_column(self, col_name: str) -> np.ndarray:
        return self._columns_cache.get(col_name)
        
    def generate_signal_vector(self, strategy: dict) -> np.ndarray:
        """
        Takes a strategy dictionary and outputs a 1D boolean array where True represents a valid signal.
        Handles both basic (rule-based) and advanced (weighted) modes.
        """
        if strategy.get('mode') == 'advanced':
            raw_threshold = strategy.get('tree_threshold', 0.5)
            direction = strategy.get('direction', 'LONG')
            
            # RELAXATION: Lower conviction floor for 15m noise
            threshold_floor = 0.51 if self.timeframe == '15m' else 0.52

            # Base model ML signal
            if direction == 'LONG':
                tree_threshold = max(threshold_floor, raw_threshold)
                tree_arr = self.get_column('tree_signal_up')
            else:
                tree_threshold = max(threshold_floor, raw_threshold)
                tree_arr = self.get_column('tree_signal_down')

            if tree_arr is None:
                # Fallback: ignore ML tree completely and rely rigidly on technical rules
                signal = np.ones(self.df_len, dtype=bool)
            else:
                signal = tree_arr > tree_threshold
                 
            # Add feature filters (from rules)
            for rule in strategy['rules']:
                feature, operator, threshold = rule
                arr = self.get_column(feature)
                if arr is None: continue
                
                if operator == '<':
                    signal &= (arr < threshold)
                elif operator == '>':
                    signal &= (arr > threshold)
            

            
        else:
            # Basic mode: rule-based
            signal = np.ones(self.df_len, dtype=bool)
            for rule in strategy['rules']:
                feature, operator, threshold = rule
                arr = self.get_column(feature)
                
                if arr is None:
                    return np.zeros(self.df_len, dtype=bool)
                    
                if operator == '<':
                    signal &= (arr < threshold)
                elif operator == '>':
                    signal &= (arr > threshold)
                    
        # Prevent lookahead bias by shifting signals
        signal = np.roll(signal, 1)
        signal[0] = False
        return signal

    def generate_signal_matrix(self, population: list) -> np.ndarray:
        """
        Takes a list of strategy dictionaries and builds a 2D boolean array.
        Shape: (num_candles, num_strategies)
        """
        num_strats = len(population)
        matrix = np.zeros((self.df_len, num_strats), dtype=bool)
        
        for i, strat in enumerate(population):
            matrix[:, i] = self.generate_signal_vector(strat)
            
        return matrix
        
if __name__ == "__main__":
    import os
    if os.path.exists("data/BTC_USDT_4h_10000_features.csv"):
        print("Testing Signal Matrix Generator...")
        df = pd.read_csv("data/BTC_USDT_4h_10000_features.csv", index_col='timestamp', parse_dates=True)
        sg = SignalGenerator(df)
        
        # Test 3 dummy strategy rulesets
        test_population = [
            {'direction': 'LONG', 'rules': [('RSI', '<', 30.0)]},
            {'direction': 'SHORT', 'rules': [('MACD_histogram', '>', 0.0), ('volume_ratio', '>', 1.5)]},
            {'mode': 'advanced', 'direction': 'LONG', 'features': ['trend_slope_20', 'RSI'], 'weights': [1.0, -0.5], 'activation_threshold': 0.0}
        ]
        
        mat = sg.generate_signal_matrix(test_population)
        print(f"Matrix Shape: {mat.shape} (Expected: {len(df)}, 3)")
        print(f"Signal Counts per Strategy: {mat.sum(axis=0)}")
    else:
        print("Feature dataset not found.")
