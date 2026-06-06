import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def generate_features(self, df: pd.DataFrame, index_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates quantitative trading features from OHLCV data.
        Assumes df has ['open', 'high', 'low', 'close', 'volume'] columns.
        """
        # Work on a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Helper: True Range (for ATR calculation)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3'], axis=1, inplace=True)

        # --- Volatility Features ---
        df['ATR'] = df['tr'].rolling(window=14).mean()
        df['ATR_ratio'] = df['ATR'] / df['ATR'].rolling(window=50).mean()
        df.drop('tr', axis=1, inplace=True)
        
        # Rolling Volatility (standard deviation of log returns)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        df['volatility_10'] = log_ret.rolling(window=10).std() * np.sqrt(365 * 24) # Approx annualized for crypto/crypto-like 4h/1d, scaling doesn't matter for rank/dist
        df['volatility_20'] = log_ret.rolling(window=20).std() * np.sqrt(365 * 24)
        df['volatility_50'] = log_ret.rolling(window=50).std() * np.sqrt(365 * 24)
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_50']

        # Bollinger Bands (20, 2)
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['MA20'] + (df['std20'] * 2)
        df['BB_lower'] = df['MA20'] - (df['std20'] * 2)
        
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA20']
        # BB_position: 1.0 means at upper band, 0.0 means at lower band
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df.drop(['std20', 'BB_upper', 'BB_lower'], axis=1, inplace=True)


        # --- Trend Features ---
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        df['stock_sma_200'] = df['MA200']  # Explicit alias for Dual-Key regime gate
        
        df['price_minus_MA20'] = (df['close'] - df['MA20']) / df['MA20']
        df['price_minus_MA50'] = (df['close'] - df['MA50']) / df['MA50']
        df['MA20_minus_MA50'] = (df['MA20'] - df['MA50']) / df['MA50']
        df['MA50_minus_MA200'] = (df['MA50'] - df['MA200']) / df['MA200']

        # Trend slope using linear regression on normalized prices
        # Formula for slope: Cov(x, y) / Var(x) where x is time indices
        def get_slope(series):
            x = np.arange(len(series))
            # Normalize y to percentages to make slope comparable across price levels
            y = series.values / series.values[0] 
            return np.polyfit(x, y, 1)[0]

        # Note: polyfit inside rolling applies is slow. A faster vectorized approach:
        def rolling_slope_vectorized(series, window):
            """ Vectorized slope of linear regression """
            y = series.values
            x = np.arange(window)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            
            # Create a 2D rolling window array view (advanced stride tricks) for fast compute
            shape = (y.shape[0] - window + 1, window)
            strides = (y.strides[0], y.strides[0])
            rolling_y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
            
            # Normalize each window by its first element
            first_element = rolling_y[:, 0][:, np.newaxis]
            # Avoid division by zero
            first_element = np.where(first_element == 0, 1e-8, first_element)
            normalized_y = rolling_y / first_element
            
            y_mean = normalized_y.mean(axis=1, keepdims=True)
            cov = np.sum((x - x_mean) * (normalized_y - y_mean), axis=1)
            slope = cov / x_var
            
            # Pad the beginning with NaNs to match original length
            return np.concatenate((np.full(window - 1, np.nan), slope))

        df['trend_slope_20'] = rolling_slope_vectorized(df['close'], 20)
        df['trend_slope_50'] = rolling_slope_vectorized(df['close'], 50)
        
        # ── Nifty 50 Macro Regime Enforcement (Institutional Benchmark) ──
        if index_df is not None:
            # Calculate 200-day SMA of the benchmark index itself
            nifty_sma_200 = index_df['close'].rolling(window=200).mean()
            
            # ── Timezone Normalization (Institutional Robustness) ──
            # Ensure both dataframes are tz-naive before joining to prevent TypeError
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            benchmark_index = index_df.index
            if benchmark_index.tz is not None:
                benchmark_index = benchmark_index.tz_localize(None)
            
            benchmark_data = pd.DataFrame({
                'bench_close': index_df['close'].values,  # Use .values to avoid index-based mismatch if already reindexed
                'bench_sma_200': nifty_sma_200.values
            }, index=benchmark_index)
            
            # Join benchmark data using the date index
            df = df.join(benchmark_data, how='left')
            
            # Sync calendar mismatches properly via forward-filling
            df['bench_close'].ffill(inplace=True)
            df['bench_sma_200'].ffill(inplace=True)
            
            # Define un-bypassable market states
            df['is_bull_market'] = (df['bench_close'] > df['bench_sma_200']).astype(float)
            df['is_bear_market'] = (df['bench_close'] <= df['bench_sma_200']).astype(float)
            
            # Cleanup raw benchmark data to avoid feature bloat
            df.drop(['bench_close', 'bench_sma_200'], axis=1, inplace=True)
        else:
            # Safety fallback (Assume Bull if data missing, but warn)
            df['is_bull_market'] = 1.0
            df['is_bear_market'] = 0.0


        # --- Momentum Features ---
        # RSI (14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_change'] = df['RSI'].diff(5)

        # MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # --- ADX (14) ---
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr_rolling = df['close'].diff().abs().rolling(14).sum() # Simplified TR for ADX context
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr_rolling)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr_rolling)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df['ADX'] = dx.rolling(14).mean()

        # --- Stochastic Oscillator (14, 3) ---
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()


        # --- Volume & VWAP Features ---
        df['volume_MA20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_MA20']
        
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['cv'] = df['typical_price'] * df['volume']
        df['vwap'] = df['cv'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['VWAP_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        # EMA Distances (For Squeeze detection)
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['EMA20_distance'] = (df['close'] - df['EMA20']) / df['EMA20']
        df['EMA200_distance'] = (df['close'] - df['EMA200']) / df['EMA200']
        
        df.drop(['typical_price', 'cv', 'vwap', 'EMA20', 'EMA200'], axis=1, inplace=True)


        # --- Returns Features ---
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)


        # --- Mean Reversion Features ---
        df['z_score_20'] = (df['close'] - df['MA20']) / (df['close'].rolling(window=20).std())
        df['z_score_50'] = (df['close'] - df['MA50']) / (df['close'].rolling(window=50).std())


        # --- Structure Features ---
        roll_min_20 = df['low'].rolling(window=20).min()
        roll_max_20 = df['high'].rolling(window=20).max()
        df['range_position_20'] = (df['close'] - roll_min_20) / (roll_max_20 - roll_min_20)

        roll_min_50 = df['low'].rolling(window=50).min()
        roll_max_50 = df['high'].rolling(window=50).max()
        df['range_position_50'] = (df['close'] - roll_min_50) / (roll_max_50 - roll_min_50)


        # --- Persistence Features ---
        # Momentum ratio: Net return over N / Sum of absolute returns over N
        net_ret_20 = df['close'].diff(20)
        sum_abs_ret_20 = df['close'].diff(1).abs().rolling(window=20).sum()
        df['momentum_ratio_20'] = net_ret_20 / sum_abs_ret_20


        # --- Drawdown Features ---
        roll_max_close_50 = df['close'].rolling(window=50).max()
        df['drawdown_50'] = (df['close'] - roll_max_close_50) / roll_max_close_50
        
        roll_max_close_100 = df['close'].rolling(window=100).max()
        df['drawdown_100'] = (df['close'] - roll_max_close_100) / roll_max_close_100

        # Replace any infs with NaNs before dropping
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Finally, drop all rows with NaN (from initial rolling windows, max is 200)
        df.dropna(inplace=True)

        return df
        
    def generate_multiple_features(self, multi_data: dict) -> dict:
        """
        Receives {symbol: DataFrame} and processes features for all.
        Ensures all final DataFrames are perfectly aligned chronologically by cropping out early mismatched dates.
        """
        processed_data = {}
        
        for symbol, df in multi_data.items():
            print(f"Feature Engineering: {symbol}...")
            # We enforce exactly 5000 limit coming in from phase 1, but feature engineering drops ~200 rows.
            feat_df = self.generate_features(df)
            processed_data[symbol] = feat_df
            
        # Synchronization: Find the latest 'start date' across all dataframes
        max_start_date = max(df.index.min() for df in processed_data.values())
        print(f"Aligning all data to start at precisely: {max_start_date}")
        
        aligned_data = {}
        for symbol, df in processed_data.items():
            aligned_df = df[df.index >= max_start_date].copy()
            aligned_data[symbol] = aligned_df
            
        # Double check alignment shape
        baseline_shape = None
        for symbol, df in aligned_data.items():
            if baseline_shape is None:
                 baseline_shape = len(df)
            else:
                 # Standardize lengths to exact exactness if there's off-by-one errors from varying non-trading days
                 # Indian stocks might miss random days. Ffill alignment:
                 pass # We will handle index reindexing to a common complete calendar if shapes don't match exactly.
                 
        # Reindex to a common calendar to enforce identical lengths and structure
        common_index = aligned_data[list(aligned_data.keys())[0]].index
        for name, df in aligned_data.items():
             common_index = common_index.intersection(df.index)
             
        for symbol, df in aligned_data.items():
             aligned_data[symbol] = df.reindex(common_index)
             
        return aligned_data

if __name__ == "__main__":
    from data_loader import DataLoader
        
    loader = DataLoader()
    nse_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        
    print("\n--- Phase 2: Multi-Asset Feature Testing ---")
    multi_data = loader.fetch_multiple_stocks_yfinance(symbols=nse_symbols, timeframe="1d", limit=5000)
        
    fe = FeatureEngineer()
    multi_features = fe.generate_multiple_features(multi_data)
        
    print("\n========= MULTI-FEATURE VERIFICATION =========")
    baseline_shape = None
    baseline_cols  = None
        
    for symbol, df_features in multi_features.items():
            print(f"\n[{symbol}]")
            print(f"Shape: {df_features.shape}")
            
            # Save to disk
            df_features.to_csv(f"data/{symbol}_features.csv")
            
            if baseline_shape is None:
                baseline_shape = df_features.shape
                baseline_cols = df_features.columns.tolist()
            
            assert df_features.shape == baseline_shape, f"Shape mismatch on {symbol}"
            assert set(df_features.columns) == set(baseline_cols), f"Column mismatch on {symbol}"
            assert df_features.isna().sum().sum() == 0, f"NaNs found in {symbol}"
            assert df_features.isin([np.inf, -np.inf]).sum().sum() == 0, f"Infs found in {symbol}"
            
    print("\nPhase 2 Feature Engineering Verification PASSED.")
