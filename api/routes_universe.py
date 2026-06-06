from flask import Blueprint, jsonify
from data_loader import DataLoader
from feature_engineer import FeatureEngineer

universe_bp = Blueprint('universe', __name__)

@universe_bp.route('/api/universe', methods=['GET'])
def get_universe():
    import os
    # Sector Mapping from DataLoader.py
    sector_map = {
        "TCS": "Large Cap IT", "INFY": "Large Cap IT", "WIPRO": "Large Cap IT", 
        "HCLTECH": "Large Cap IT", "TECHM": "Large Cap IT",
        "HDFCBANK": "Private Banks", "ICICIBANK": "Private Banks", 
        "AXISBANK": "Private Banks", "KOTAKBANK": "Private Banks", "INDUSINDBK": "Private Banks",
        "RELIANCE": "Energy / Conglomerate", "ONGC": "Energy / Conglomerate", 
        "NTPC": "Energy / Conglomerate", "POWERGRID": "Energy / Conglomerate", "COALINDIA": "Energy / Conglomerate",
        "TATASTEEL": "Metals / Materials", "JSWSTEEL": "Metals / Materials", 
        "HINDALCO": "Metals / Materials", "VEDL": "Metals / Materials", "SAIL": "Metals / Materials",
        "HINDUNILVR": "FMCG / Consumer", "ITC": "FMCG / Consumer", 
        "NESTLEIND": "FMCG / Consumer", "BRITANNIA": "FMCG / Consumer", "DABUR": "FMCG / Consumer",
        "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", 
        "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma"
    }

    stocks = []
    # Always include Crypto
    stocks.append({
        "symbol": "BTC-USD",
        "full_name": "Bitcoin USD",
        "sector": "Cryptocurrency",
        "exchange": "Crypto"
    })
    
    kite_data_dir = os.path.join("data", "nifty50")
    if os.path.exists(kite_data_dir):
        for f in sorted(os.listdir(kite_data_dir)):
            if f.endswith('_day.csv'):
                sym = f.replace('_day.csv', '')
                sector = sector_map.get(sym, "Large Cap Indian Equity")
                stocks.append({
                    "symbol": sym,
                    "full_name": f"{sym} Ltd",
                    "sector": sector,
                    "exchange": "NSE"
                })
    else:
        # Fallback if specific folder doesn't exist yet
        for sym, sector in sector_map.items():
            stocks.append({
                "symbol": sym,
                "full_name": f"{sym} Ltd",
                "sector": sector,
                "exchange": "NSE"
            })
    
    return jsonify({"stocks": stocks})

@universe_bp.route('/api/features', methods=['GET'])
def get_features():
    feature_groups = {
        "Mean Reversion": ["z_score_20", "z_score_50", "range_position_20", "range_position_50"],
        "Trend":          ["trend_slope_20", "trend_slope_50", "MACD_histogram"],
        "Volatility":     ["BB_width", "BB_position", "volatility_ratio", "ATR_ratio"],
        "Oscillator":     ["RSI", "RSI_change"],
        "Volume":         ["volume_ratio", "VWAP_distance"],
        "Momentum":       ["return_5", "return_20", "momentum_ratio_20", "drawdown_50", "drawdown_100"]
    }
    return jsonify({"feature_groups": feature_groups})
