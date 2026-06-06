import random
from collections import Counter

class StrategyGenerator:
    def __init__(self):
        # Precise list of 16 features as requested by user
        self.features = {
            'RSI': (10.0, 90.0),
            'RSI_change': (-50.0, 50.0),
            'MACD_histogram': (-2.0, 2.0),
            'trend_slope_20': (-0.02, 0.02),
            'trend_slope_50': (-0.02, 0.02),
            'trend_slope_50': (-0.02, 0.02),
            'volatility_ratio': (0.5, 2.0),
            'ATR_ratio': (0.5, 2.0),
            'BB_position': (0.0, 1.0),
            'BB_width': (0.0, 0.3),
            'volume_ratio': (0.5, 3.0),
            'VWAP_distance': (-0.1, 0.1),
            'return_5': (-0.2, 0.2),
            'return_20': (-0.4, 0.4),
            'z_score_20': (-3.0, 3.0),
            'z_score_50': (-3.0, 3.0),
            'range_position_20': (0.0, 1.0),
            'range_position_50': (0.0, 1.0),
            'momentum_ratio_20': (0.0, 1.0),
            'drawdown_50': (-1.0, 0.0),
            'drawdown_100': (-1.0, 0.0),
            'ADX': (0.0, 60.0),
            'EMA20_distance': (-0.15, 0.15),
            'EMA200_distance': (-0.30, 0.30)
        }
        
        # Grouped features for diversity
        self.feature_groups = {
            'A': ['z_score_20', 'z_score_50', 'range_position_20', 'range_position_50', 'EMA20_distance', 'VWAP_distance'],
            'B': ['trend_slope_20', 'trend_slope_50', 'ADX', 'momentum_ratio_20'],
            'C': ['BB_width', 'BB_position', 'volatility_ratio', 'ATR_ratio'],
            'D': ['return_5', 'return_20', 'RSI_change', 'RSI', 'MACD_histogram', 
                  'volume_ratio', 'drawdown_50', 'drawdown_100', 'EMA200_distance']
        }
        
    def generate_random_strategy(self, regime_name: str = "Trend", forced_direction: str = None, timeframe: str = '1d') -> dict:
        """ Generates a random set of rules filtered by regime-specific pools. """
        direction = forced_direction if forced_direction else random.choice(['LONG', 'SHORT'])
        
        # ── REGIME-SPECIFIC POOLS (Robustness Upgrade) ────────────────
        if regime_name == "Trend":
            pool = self.feature_groups['B'] + self.feature_groups['C']
        elif regime_name == "MeanRev":
            pool = self.feature_groups['A'] + self.feature_groups['D']
        elif regime_name == "Squeeze":
            pool = self.feature_groups['C'] + ['ADX', 'volume_ratio']
        else:
            pool = list(self.features.keys())
        # Ensure we only pick features that actually exist in the features dictionary (which can be customized at runtime)
        pool = [f for f in pool if f in self.features]
        if not pool:
            pool = list(self.features.keys())
            
        num_rules = random.randint(1, 3)
        selected_features = random.sample(pool, min(num_rules, len(pool)))
        
        rules = []
        for feature in selected_features:
            min_val, max_val = self.features[feature]
            threshold = random.uniform(min_val, max_val)
            operator = random.choice(['<', '>'])
            round_digits = 4 if 'trend_slope' in feature or 'VWAP_distance' in feature else 2
            rules.append((feature, operator, round(threshold, round_digits)))
            
        return {
            'direction': direction,
            'rules': rules
        }
        
    def generate_population(self, size=200, advanced_mode: bool = False, allowed_features: list = None, forced_direction: str = None, timeframe: str = '1d') -> list:
        return [self.generate_random_strategy(advanced_mode, allowed_features, forced_direction, timeframe) for _ in range(size)]

if __name__ == "__main__":
    generator = StrategyGenerator()
    
    print("========= STRATEGY GENERATION (POPULATION OF 200) =========")
    population = generator.generate_population(200)
    
    # Validation stats
    directions = [s['direction'] for s in population]
    rule_counts = [len(s['rules']) for s in population]
    
    dir_counts = Counter(directions)
    rule_len_counts = Counter(rule_counts)
    
    print("\n--- Distribution of Directions ---")
    for d, count in dir_counts.items():
        print(f"{d}: {count} ({count/200:.1%})")
        
    print("\n--- Distribution of Rule Counts ---")
    for r, count in sorted(rule_len_counts.items()):
        print(f"{r} rule(s): {count} ({count/200:.1%})")
        
    print("\n--- 5 Example Strategies ---")
    for i in range(5):
        strat = population[i]
        print(f"Strategy {i+1}:")
        print(f"  direction = {strat['direction']}")
        print(f"  entry_rules:")
        for rule in strat['rules']:
            print(f"    {rule}")
        print()
