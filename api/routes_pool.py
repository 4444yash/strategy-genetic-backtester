from flask import Blueprint, jsonify, request
import json
import os
import yfinance as yf
import pandas as pd

pool_bp = Blueprint('pool', __name__)

@pool_bp.route('/api/pool', methods=['GET'])
def get_pool():
    if not os.path.exists('strategy_pool.json'):
        return jsonify({"strategies": []})
    
    with open('strategy_pool.json', 'r') as f:
        try:
            pool = json.load(f)
        except:
            pool = []
    
    return jsonify({"strategies": pool})

@pool_bp.route('/api/benchmark', methods=['POST'])
def get_benchmark():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    if not start_date or not end_date:
        return jsonify({"error": "Start and End dates required"}), 400

    try:
        nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty.columns = [c.lower() for c in nifty.columns]
        
        # Ensure 'close' exists
        if 'close' not in nifty.columns:
            return jsonify({"error": "No benchmark data found"}), 404
        
        nifty = nifty['close']
        
        # Normalize to 100
        norm_values = (nifty / nifty.iloc[0] * 100).round(2).tolist()
        dates = nifty.index.strftime('%Y-%m-%d').tolist()
        
        return jsonify({"dates": dates, "values": norm_values})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
