import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class GameMathematics:
    """Comprehensive toolkit for online gambling game mathematics"""
    
    @staticmethod
    def calculate_ev(probabilities: List[float], payouts: List[float]) -> float:
        """
        Calculate Expected Value
        
        Args:
            probabilities: List of probabilities for each outcome
            payouts: List of payouts for each outcome (net profit)
        
        Returns:
            Expected value per bet unit
        """
        return sum(p * payout for p, payout in zip(probabilities, payouts))
    
    @staticmethod
    def calculate_rtp(probabilities: List[float], payouts: List[float], bet: float = 1.0) -> float:
        """
        Calculate Return to Player percentage
        
        Args:
            probabilities: List of probabilities for each outcome
            payouts: List of total returns for each outcome
            bet: Bet amount (default 1.0)
        
        Returns:
            RTP as percentage
        """
        expected_return = sum(p * payout for p, payout in zip(probabilities, payouts))
        return (expected_return / bet) * 100
    
    @staticmethod
    def calculate_house_edge(rtp: float) -> float:
        """Calculate house edge from RTP"""
        return 100 - rtp
    
    @staticmethod
    def calculate_variance(probabilities: List[float], payouts: List[float], ev: float = None) -> float:
        """
        Calculate variance of a game
        
        Args:
            probabilities: List of probabilities
            payouts: List of payouts
            ev: Expected value (calculated if not provided)
        
        Returns:
            Variance
        """
        if ev is None:
            ev = GameMathematics.calculate_ev(probabilities, payouts)
        
        variance = sum(p * (payout - ev) ** 2 for p, payout in zip(probabilities, payouts))
        return variance
    
    @staticmethod
    def calculate_std_dev(variance: float) -> float:
        """Calculate standard deviation from variance"""
        return np.sqrt(variance)
    
    @staticmethod
    def calculate_hit_frequency(probabilities: List[float], winning_indices: List[int]) -> float:
        """
        Calculate hit frequency (percentage of winning rounds)
        
        Args:
            probabilities: All outcome probabilities
            winning_indices: Indices of winning outcomes
        
        Returns:
            Hit frequency as percentage
        """
        return sum(probabilities[i] for i in winning_indices) * 100
    
    @staticmethod
    def simulate_game(n_rounds: int, probabilities: List[float], 
                     payouts: List[float], bet: float = 1.0) -> Dict:
        """
        Monte Carlo simulation of game outcomes
        
        Args:
            n_rounds: Number of rounds to simulate
            probabilities: Outcome probabilities
            payouts: Net payouts for each outcome
            bet: Bet amount per round
        
        Returns:
            Dictionary with simulation results
        """
        outcomes = np.random.choice(len(probabilities), size=n_rounds, p=probabilities)
        results = [payouts[outcome] * bet for outcome in outcomes]
        
        cumulative = np.cumsum(results)
        
        return {
            'total_wagered': n_rounds * bet,
            'total_returned': sum(results) + (n_rounds * bet),
            'net_profit': sum(results),
            'actual_rtp': ((sum(results) + (n_rounds * bet)) / (n_rounds * bet)) * 100,
            'results': results,
            'cumulative': cumulative,
            'final_balance': cumulative[-1] if len(cumulative) > 0 else 0
        }
    
    @staticmethod
    def calculate_volatility_index(std_dev: float, ev: float) -> float:
        """
        Calculate volatility index (coefficient of variation)
        Higher values = more volatile game
        """
        if ev == 0:
            return float('inf')
        return abs(std_dev / ev)
    
    @staticmethod
    def bankroll_requirement(rtp: float, std_dev: float, 
                            n_rounds: int, confidence: float = 0.95) -> float:
        """
        Calculate recommended bankroll to survive n rounds with given confidence
        
        Args:
            rtp: Return to player percentage
            std_dev: Standard deviation per round
            n_rounds: Number of rounds to survive
            confidence: Confidence level (default 95%)
        
        Returns:
            Recommended bankroll in bet units
        """
        z_score = stats.norm.ppf(confidence)
        expected_loss = n_rounds * (1 - rtp/100)
        volatility = std_dev * np.sqrt(n_rounds)
        return expected_loss + (z_score * volatility)


class SlotMathematics:
    """Specialized mathematics for slot games"""
    
    @staticmethod
    def calculate_combinations(reel_strips: List[List[str]]) -> int:
        """Calculate total possible combinations"""
        return np.prod([len(reel) for reel in reel_strips])
    
    @staticmethod
    def calculate_symbol_probability(reel_strips: List[List[str]], 
                                     symbol: str) -> List[float]:
        """Calculate probability of symbol on each reel"""
        return [reel.count(symbol) / len(reel) for reel in reel_strips]
    
    @staticmethod
    def calculate_combination_probability(reel_strips: List[List[str]], 
                                         symbols: List[str]) -> float:
        """Calculate probability of specific symbol combination"""
        prob = 1.0
        for reel, symbol in zip(reel_strips, symbols):
            prob *= reel.count(symbol) / len(reel)
        return prob
    
    @staticmethod
    def build_paytable_analysis(reel_strips: List[List[str]], 
                                paytable: Dict[Tuple[str, ...], float]) -> pd.DataFrame:
        """
        Analyze complete paytable
        
        Args:
            reel_strips: List of reel strip configurations
            paytable: Dictionary mapping symbol combinations to payouts
        
        Returns:
            DataFrame with analysis of each winning combination
        """
        results = []
        
        for combo, payout in paytable.items():
            prob = SlotMathematics.calculate_combination_probability(reel_strips, list(combo))
            contribution = prob * payout
            
            results.append({
                'combination': combo,
                'payout': payout,
                'probability': prob,
                'frequency': f"1 in {int(1/prob) if prob > 0 else 'inf'}",
                'ev_contribution': contribution
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('ev_contribution', ascending=False)


class RouletteAnalyzer:
    """Mathematics for roulette games"""
    
    @staticmethod
    def european_roulette_ev(bet_type: str, payout: float) -> Dict:
        """
        Calculate EV for European Roulette (37 numbers: 0-36)
        
        Common bet types and their winning numbers:
        - straight: 1 number (payout 35:1)
        - split: 2 numbers (payout 17:1)
        - street: 3 numbers (payout 11:1)
        - corner: 4 numbers (payout 8:1)
        - line: 6 numbers (payout 5:1)
        - dozen: 12 numbers (payout 2:1)
        - column: 12 numbers (payout 2:1)
        - red/black: 18 numbers (payout 1:1)
        - even/odd: 18 numbers (payout 1:1)
        - high/low: 18 numbers (payout 1:1)
        """
        bet_configs = {
            'straight': 1, 'split': 2, 'street': 3, 'corner': 4,
            'line': 6, 'dozen': 12, 'column': 12,
            'red': 18, 'black': 18, 'even': 18, 'odd': 18,
            'high': 18, 'low': 18
        }
        
        winning_numbers = bet_configs.get(bet_type, 0)
        total_numbers = 37
        
        prob_win = winning_numbers / total_numbers
        prob_lose = 1 - prob_win
        
        ev = (prob_win * payout) + (prob_lose * -1)
        rtp = ((prob_win * (payout + 1)) / 1) * 100
        
        return {
            'bet_type': bet_type,
            'probability_win': prob_win,
            'payout': payout,
            'expected_value': ev,
            'rtp': rtp,
            'house_edge': 100 - rtp
        }


# Example usage and demonstrations
if __name__ == "__main__":
    print("=== GAME MATHEMATICS TOOLKIT ===\n")
    
    # Example 1: Simple game analysis
    print("Example 1: Coin Flip Game (Fair)")
    print("-" * 50)
    probs = [0.5, 0.5]
    payouts = [1, -1]  # Win $1 or lose $1
    
    ev = GameMathematics.calculate_ev(probs, payouts)
    variance = GameMathematics.calculate_variance(probs, payouts, ev)
    std_dev = GameMathematics.calculate_std_dev(variance)
    
    print(f"Expected Value: ${ev:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Volatility Index: {GameMathematics.calculate_volatility_index(std_dev, ev):.4f}")
    
    # Example 2: Slot-style game
    print("\n\nExample 2: Simple Slot Game")
    print("-" * 50)
    # Probabilities: Big Win, Small Win, Loss
    probs = [0.01, 0.15, 0.84]
    payouts = [100, 2, -1]  # Net payouts
    
    ev = GameMathematics.calculate_ev(probs, payouts)
    rtp = GameMathematics.calculate_rtp(probs, [p + 1 for p in payouts], bet=1.0)
    variance = GameMathematics.calculate_variance(probs, payouts, ev)
    std_dev = GameMathematics.calculate_std_dev(variance)
    hit_freq = GameMathematics.calculate_hit_frequency(probs, [0, 1])
    
    print(f"RTP: {rtp:.2f}%")
    print(f"House Edge: {GameMathematics.calculate_house_edge(rtp):.2f}%")
    print(f"Expected Value: ${ev:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Hit Frequency: {hit_freq:.2f}%")
    
    # Simulation
    sim_results = GameMathematics.simulate_game(10000, probs, payouts, bet=1.0)
    print(f"\nSimulation (10,000 rounds):")
    print(f"Actual RTP: {sim_results['actual_rtp']:.2f}%")
    print(f"Net Profit/Loss: ${sim_results['net_profit']:.2f}")
    
    # Example 3: European Roulette
    print("\n\nExample 3: European Roulette - Red/Black Bet")
    print("-" * 50)
    roulette = RouletteAnalyzer.european_roulette_ev('red', 1)
    for key, value in roulette.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Example 4: Bankroll management
    print("\n\nExample 4: Bankroll Recommendation")
    print("-" * 50)
    bankroll = GameMathematics.bankroll_requirement(
        rtp=96.5, 
        std_dev=10.0, 
        n_rounds=1000, 
        confidence=0.95
    )
    print(f"Recommended bankroll for 1000 rounds at 95% confidence:")
    print(f"{bankroll:.2f} bet units")
    print(f"For $1 bets: ${bankroll:.2f}")
