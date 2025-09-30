import numpy as np
import pandas as pd
from itertools import product
from collections import Counter

class SlotGameDesigner:
    """Complete slot game design and mathematical analysis"""
    
    def __init__(self, reel_strips, paytable, bet_amount=1.0):
        """
        Initialize slot game
        
        Args:
            reel_strips: List of lists, each inner list represents a reel
            paytable: Dict with winning combinations and their payouts
            bet_amount: Base bet amount
        """
        self.reel_strips = reel_strips
        self.paytable = paytable
        self.bet_amount = bet_amount
        self.analysis = None
    
    def get_total_combinations(self):
        """Calculate total possible outcomes"""
        return np.prod([len(reel) for reel in self.reel_strips])
    
    def get_symbol_counts(self):
        """Get symbol distribution across all reels"""
        all_symbols = []
        for reel in self.reel_strips:
            all_symbols.extend(reel)
        return Counter(all_symbols)
    
    def calculate_combination_probability(self, symbols):
        """Calculate probability of specific symbol combination appearing"""
        prob = 1.0
        for reel_idx, symbol in enumerate(symbols):
            reel = self.reel_strips[reel_idx]
            prob *= reel.count(symbol) / len(reel)
        return prob
    
    def analyze_all_outcomes(self):
        """Comprehensive analysis of all possible outcomes"""
        total_combos = self.get_total_combinations()
        
        # Generate all possible combinations
        all_combinations = list(product(*self.reel_strips))
        
        results = []
        total_ev_contribution = 0
        
        for combo in all_combinations:
            # Check if this combination wins
            payout = 0
            winning_combo = None
            
            for pay_combo, pay_amount in self.paytable.items():
                if self._matches_pattern(combo, pay_combo):
                    if pay_amount > payout:  # Take highest payout if multiple matches
                        payout = pay_amount
                        winning_combo = pay_combo
            
            prob = 1 / total_combos
            ev_contribution = prob * payout
            total_ev_contribution += ev_contribution
            
            if payout > 0:  # Only store winning combinations
                results.append({
                    'combination': combo,
                    'pattern': winning_combo,
                    'payout': payout,
                    'probability': prob,
                    'frequency': f"1 in {int(1/prob)}",
                    'ev_contribution': ev_contribution
                })
        
        # Create summary
        df = pd.DataFrame(results) if results else pd.DataFrame()
        
        self.analysis = {
            'total_combinations': total_combos,
            'winning_combinations': len(results),
            'total_ev': total_ev_contribution,
            'rtp': (total_ev_contribution / self.bet_amount) * 100,
            'house_edge': 100 - ((total_ev_contribution / self.bet_amount) * 100),
            'details': df
        }
        
        return self.analysis
    
    def _matches_pattern(self, combo, pattern):
        """Check if combination matches winning pattern"""
        # Support for wildcards (represented as '*' or 'WILD')
        for i, (symbol, pattern_symbol) in enumerate(zip(combo, pattern)):
            if pattern_symbol == '*' or pattern_symbol == 'WILD':
                continue
            if symbol == 'WILD':  # Wild substitutes for any symbol
                continue
            if symbol != pattern_symbol:
                return False
        return True
    
    def calculate_variance(self):
        """Calculate game variance"""
        if self.analysis is None:
            self.analyze_all_outcomes()
        
        total_combos = self.analysis['total_combinations']
        ev = self.analysis['total_ev']
        
        # Calculate variance across all outcomes
        variance = 0
        for _, row in self.analysis['details'].iterrows():
            deviation = (row['payout'] - ev) ** 2
            variance += row['probability'] * deviation
        
        # Add losing combinations
        losing_prob = 1 - self.analysis['details']['probability'].sum()
        if losing_prob > 0:
            variance += losing_prob * (0 - ev) ** 2
        
        return variance
    
    def calculate_hit_frequency(self):
        """Calculate percentage of spins that result in a win"""
        if self.analysis is None:
            self.analyze_all_outcomes()
        
        if self.analysis['details'].empty:
            return 0.0
        
        return self.analysis['details']['probability'].sum() * 100
    
    def simulate_game_session(self, n_spins=10000, starting_balance=1000):
        """Simulate a gaming session"""
        balance = starting_balance
        results = []
        
        for _ in range(n_spins):
            if balance < self.bet_amount:
                break
            
            balance -= self.bet_amount
            
            # Spin reels
            outcome = tuple(reel[np.random.randint(len(reel))] 
                          for reel in self.reel_strips)
            
            # Check for win
            payout = 0
            for pay_combo, pay_amount in self.paytable.items():
                if self._matches_pattern(outcome, pay_combo):
                    payout = max(payout, pay_amount)
            
            balance += payout
            results.append({
                'spin': len(results) + 1,
                'outcome': outcome,
                'payout': payout,
                'balance': balance
            })
        
        df = pd.DataFrame(results)
        
        return {
            'final_balance': balance,
            'total_wagered': len(results) * self.bet_amount,
            'total_won': df['payout'].sum(),
            'net_profit': balance - starting_balance,
            'actual_rtp': (df['payout'].sum() / (len(results) * self.bet_amount)) * 100,
            'spins_played': len(results),
            'winning_spins': len(df[df['payout'] > 0]),
            'actual_hit_frequency': (len(df[df['payout'] > 0]) / len(results)) * 100,
            'history': df
        }
    
    def print_summary(self):
        """Print comprehensive game summary"""
        if self.analysis is None:
            self.analyze_all_outcomes()
        
        print("=" * 70)
        print("SLOT GAME MATHEMATICAL ANALYSIS")
        print("=" * 70)
        
        print(f"\nGAME CONFIGURATION:")
        print(f"  Number of reels: {len(self.reel_strips)}")
        print(f"  Reel lengths: {[len(reel) for reel in self.reel_strips]}")
        print(f"  Total combinations: {self.analysis['total_combinations']:,}")
        print(f"  Winning combinations: {self.analysis['winning_combinations']:,}")
        
        print(f"\nKEY METRICS:")
        print(f"  RTP: {self.analysis['rtp']:.4f}%")
        print(f"  House Edge: {self.analysis['house_edge']:.4f}%")
        print(f"  Expected Value per spin: ${self.analysis['total_ev']:.6f}")
        print(f"  Hit Frequency: {self.calculate_hit_frequency():.4f}%")
        
        variance = self.calculate_variance()
        std_dev = np.sqrt(variance)
        print(f"  Variance: {variance:.4f}")
        print(f"  Standard Deviation: {std_dev:.4f}")
        print(f"  Volatility: {'High' if std_dev > 10 else 'Medium' if std_dev > 5 else 'Low'}")
        
        print(f"\nTOP PAYING COMBINATIONS:")
        if not self.analysis['details'].empty:
            top_pays = self.analysis['details'].nlargest(10, 'payout')
            for idx, row in top_pays.iterrows():
                print(f"  {row['pattern']}: {row['payout']}x "
                      f"(Freq: {row['frequency']}, "
                      f"EV Contrib: {row['ev_contribution']:.6f})")
        
        print("\n" + "=" * 70)


# Example: Build a classic 3-reel slot game
if __name__ == "__main__":
    
    # Define reel strips (what symbols appear on each reel)
    reel_strips = [
        ['7', '7', 'BAR', 'BAR', 'BAR', 'CHERRY', 'CHERRY', 'CHERRY', 
         'LEMON', 'LEMON', 'ORANGE', 'ORANGE', 'PLUM', 'PLUM', 'BELL', 'BELL'],
        
        ['7', 'BAR', 'BAR', 'BAR', 'CHERRY', 'CHERRY', 'CHERRY', 'CHERRY',
         'LEMON', 'LEMON', 'ORANGE', 'ORANGE', 'PLUM', 'PLUM', 'BELL', 'BELL'],
        
        ['7', 'BAR', 'BAR', 'CHERRY', 'CHERRY', 'CHERRY', 'CHERRY', 'CHERRY',
         'LEMON', 'LEMON', 'LEMON', 'ORANGE', 'ORANGE', 'PLUM', 'BELL', 'BELL']
    ]
    
    # Define paytable (winning combinations and their payouts)
    # Format: (reel1_symbol, reel2_symbol, reel3_symbol): payout_multiplier
    paytable = {
        ('7', '7', '7'): 100,           # Jackpot
        ('BAR', 'BAR', 'BAR'): 20,      # Three BARs
        ('BELL', 'BELL', 'BELL'): 15,   # Three BELLs
        ('PLUM', 'PLUM', 'PLUM'): 10,   # Three PLUMs
        ('ORANGE', 'ORANGE', 'ORANGE'): 8,  # Three ORANGEs
        ('LEMON', 'LEMON', 'LEMON'): 6,    # Three LEMONs
        ('CHERRY', 'CHERRY', 'CHERRY'): 5, # Three CHERRYs
        ('CHERRY', 'CHERRY', '*'): 2,      # Two CHERRYs (any third symbol)
        ('CHERRY', '*', '*'): 1,           # One CHERRY (any other symbols)
    }
    
    # Create and analyze the game
    slot = SlotGameDesigner(reel_strips, paytable, bet_amount=1.0)
    slot.print_summary()
    
    # Run simulation
    print("\nRUNNING SIMULATION (10,000 spins)...")
    sim = slot.simulate_game_session(n_spins=10000, starting_balance=1000)
    
    print(f"\nSIMULATION RESULTS:")
    print(f"  Spins played: {sim['spins_played']:,}")
    print(f"  Total wagered: ${sim['total_wagered']:.2f}")
    print(f"  Total won: ${sim['total_won']:.2f}")
    print(f"  Net profit/loss: ${sim['net_profit']:.2f}")
    print(f"  Final balance: ${sim['final_balance']:.2f}")
    print(f"  Actual RTP: {sim['actual_rtp']:.4f}%")
    print(f"  Actual hit frequency: {sim['actual_hit_frequency']:.4f}%")
    print(f"  Winning spins: {sim['winning_spins']:,}")
