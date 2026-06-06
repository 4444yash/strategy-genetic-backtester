import pandas as pd
import ccxt
import yfinance as yf
import os
import time

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def validate_data(self, df):
        if df.empty:
            return df
        
        # Ensure timestamp is index
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            
        # 1. Remove duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort index just in case
        df = df.sort_index()

        # 2. Handle missing candles & 3. Remove NaN values
        # Forward fill any missing gaps, then drop remaining NaNs (usually from the start)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def fetch_crypto_ccxt(self, symbol="BTC/USDT", timeframe="4h", limit=10000):
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{limit}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath):
            print(f"Loading {symbol} from local CSV: {filepath}")
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            if len(df) > limit:
                df = df.iloc[-limit:]
            return df
            
        print(f"Fetching {symbol} from Binance via CCXT... This might take a moment due to pagination.")
        exchange = ccxt.binance({'enableRateLimit': True})
        
        try:
            tf_ms = exchange.parse_timeframe(timeframe) * 1000
        except Exception:
            print(f"Unsupported timeframe for ccxt: {timeframe}")
            return pd.DataFrame()
            
        now = exchange.milliseconds()
        start_time = now - (limit * tf_ms)
        
        all_ohlcv = []
        current_since = start_time
        
        while len(all_ohlcv) < limit:
            fetch_limit = min(1000, limit - len(all_ohlcv))
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit)
                if not ohlcv or len(ohlcv) == 0:
                    break
                    
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + tf_ms
                
                # Sleep a bit to respect rate limits
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
                
        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df = self.validate_data(df)
        
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        df.to_csv(filepath)
        print(f"Saved to {filepath}")
        
        return df

    def fetch_stocks_yfinance(self, symbol="AAPL", timeframe="1d", limit=10000, sector="Unknown"):
        filename = f"{symbol}_{timeframe}_{limit}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath):
            print(f"Loading {symbol} from local CSV: {filepath}")
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            if 'sector' not in df.columns:
                df['sector'] = sector
                df.to_csv(filepath)
            if len(df) > limit:
                df = df.iloc[-limit:]
            return df
            
        print(f"Fetching {symbol} from Yahoo Finance...")
        
        tf_map = {
            "1d": "1d",
            "1h": "1h",
            "4h": "1h", # Fallback 
            "1m": "1m"
        }
        yf_tf = tf_map.get(timeframe, timeframe)
        
        period = "max"
        if yf_tf.endswith("h") or yf_tf.endswith("m"):
             period = "730d" if yf_tf == "1h" else "60d" 
             
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=yf_tf)
                if df.empty:
                   df = yf.download(symbol, period=period, interval=yf_tf, progress=False)
                if not df.empty:
                    break
            except Exception as e:
                print(f"Error fetching data attempt {attempt+1}: {e}")
                time.sleep(2)
                
        if df.empty:
            print(f"Warning: Failed to fetch data for {symbol} after retries.")
            return pd.DataFrame()
            
        # Standardize YFinance format
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, axis=1, level=1, drop_level=True) if symbol in df.columns.levels[1] else df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

        df.columns = [c.lower() for c in df.columns]
        
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"Missing columns from yfinance: {missing}")
            return pd.DataFrame()
            
        df = df[req_cols].copy()
        df.index.name = 'timestamp'
        
        if df.index.tz is not None:
             df.index = df.index.tz_convert('UTC').tz_localize(None)
             
        if timeframe == "4h" and yf_tf == "1h":
             df = df.resample('4h').agg({
                 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
             }).dropna()
             
        df = self.validate_data(df)
        
        # Add sector
        df['sector'] = sector
        
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        df.to_csv(filepath)
        print(f"Saved to {filepath}")
        
        return df
        
    def load_local_kite_data(self, symbol="RELIANCE"):
        filepath = os.path.join(self.data_dir, "nifty50", f"{symbol}_day.csv")
        if os.path.exists(filepath):
            print(f"Loading {symbol} from local Kite CSV: {filepath}")
            # Kite headers: date, open, high, low, close, volume
            df = pd.read_csv(filepath, index_col='date', parse_dates=True)
            df.index.name = 'timestamp'
            df = self.validate_data(df)
            return df
        else:
            print(f"Warning: Local dataset for {symbol} not found. Attempting YFinance fallback.")
            return self.fetch_stocks_yfinance(symbol=symbol+".NS", timeframe="1d", limit=3200)

    def load_local_kite_15m(self, symbol="RELIANCE"):
        filepath = os.path.join(self.data_dir, "nifty15m", f"{symbol}_15m.csv")
        if os.path.exists(filepath):
            print(f"Loading {symbol} from local Kite Intraday CSV: {filepath}")
            # Kite headers: date, open, high, low, close, volume
            df = pd.read_csv(filepath, index_col='date', parse_dates=True)
            df.index.name = 'timestamp'
            df = self.validate_data(df)
            return df
        else:
            print(f"Warning: Local 15m dataset for {symbol} not found. Attempting YFinance fallback.")
            # Yfinance limited to 60 days for 15m
            return self.fetch_stocks_yfinance(symbol=symbol+".NS", timeframe="15m", limit=3000)

    def load_nifty_index(self, timeframe="1d"):
        if timeframe == "15m":
            filepath = os.path.join(self.data_dir, "nifty15m", "NIFTY50_15m.csv")
            if os.path.exists(filepath):
                print(f"Loading NIFTY 50 (15m) from local CSV: {filepath}")
                df = pd.read_csv(filepath, index_col='date', parse_dates=True)
                df.index.name = 'timestamp'
                df = self.validate_data(df)
                return df
            else:
                print(f"[CRITICAL WARNING] Nifty 50 15m (macro index) not found at {filepath}. Resorting to YFinance fallback. Expect potential alignment drift.")
                return self.fetch_stocks_yfinance(symbol="^NSEI", timeframe="15m", limit=5000)
        
        # Default Daily
        filepath = os.path.join(self.data_dir, "nifty50", "NIFTY_50_day.csv")
        if os.path.exists(filepath):
            print(f"Loading NIFTY 50 index from local CSV: {filepath}")
            df = pd.read_csv(filepath, index_col='date', parse_dates=True)
            df.index.name = 'timestamp'
            df = self.validate_data(df)
            return df
        else:
            print("[WARNING] NIFTY 50 index file not found. Falling back to YFinance.")
            return self.fetch_stocks_yfinance(symbol="^NSEI", timeframe="1d", limit=5000)

    def fetch_multiple_stocks_yfinance(self, symbols: list = None, timeframe="1d", limit=10000) -> dict:
        """
        Fetches data for multiple stocks and returns a dictionary of DataFrames.
        Data is standardized chronologically and cleaned.
        """
        data_dict = {}
        
        # Sector Mapping
        sector_map = {
            "TCS.NS": "Large Cap IT", "INFY.NS": "Large Cap IT", "WIPRO.NS": "Large Cap IT", "HCLTECH.NS": "Large Cap IT", "TECHM.NS": "Large Cap IT",
            "HDFCBANK.NS": "Private Banks", "ICICIBANK.NS": "Private Banks", "AXISBANK.NS": "Private Banks", "KOTAKBANK.NS": "Private Banks", "INDUSINDBK.NS": "Private Banks",
            "RELIANCE.NS": "Energy / Conglomerate", "ONGC.NS": "Energy / Conglomerate", "NTPC.NS": "Energy / Conglomerate", "POWERGRID.NS": "Energy / Conglomerate", "COALINDIA.NS": "Energy / Conglomerate",
            "TATASTEEL.NS": "Metals / Materials", "JSWSTEEL.NS": "Metals / Materials", "HINDALCO.NS": "Metals / Materials", "VEDL.NS": "Metals / Materials", "SAIL.NS": "Metals / Materials",
            "HINDUNILVR.NS": "FMCG / Consumer", "ITC.NS": "FMCG / Consumer", "NESTLEIND.NS": "FMCG / Consumer", "BRITANNIA.NS": "FMCG / Consumer", "DABUR.NS": "FMCG / Consumer",
            "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma", "APOLLOHOSP.NS": "Pharma"
        }
        
        if symbols is None:
            symbols = list(sector_map.keys())
            
        for symbol in symbols:
            sector = sector_map.get(symbol, "Unknown")
            df = self.fetch_stocks_yfinance(symbol=symbol, timeframe=timeframe, limit=limit, sector=sector)
            
            if df.empty:
                print(f"Warning: No valid data retrieved for {symbol}")
                continue
                
            # Data quality check
            if len(df) < 500:
                print(f"Warning: {symbol} has {len(df)} bars (< 500). Skipping due to insufficient history.")
                continue
                
            base_name = symbol.split('.')[0]
            data_dict[base_name] = df
                
        return data_dict

if __name__ == "__main__":
    loader = DataLoader()
    print("\n--- Fetching Multi-Stock NSE Data (30 Stocks) ---")
    multi_data = loader.fetch_multiple_stocks_yfinance(timeframe="1d", limit=5000)
    print(f"\nSuccessfully loaded {len(multi_data)} stocks.")
