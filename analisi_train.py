#!/usr/bin/env python3
"""
Enhanced RCT-RL Training Analyzer
Analisi completa e dettagliata di training RL con report professionale
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedRCTRLAnalyzer:
    def __init__(self, results_dir=None, output_dir=None):
        """
        Args:
            results_dir: Path to ray_results/PPO_* directory
            output_dir: Where to save analysis (default: results_dir/analysis)
        """
        if results_dir is None:
            results_dir = self._find_latest_run()
        
        self.results_dir = Path(results_dir).expanduser()
        
        if output_dir is None:
            self.output_dir = self.results_dir / "analysis"
        else:
            self.output_dir = Path(output_dir).expanduser()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find progress.csv
        self.csv_path = self._find_progress_csv()
        
        # Load config if available
        self.config = self._load_config()
        
    def _find_latest_run(self):
        """Auto-detect latest Ray results directory"""
        ray_results = Path("~/ray_results").expanduser()
        if not ray_results.exists():
            raise FileNotFoundError("~/ray_results not found")
        
        runs = sorted(ray_results.glob("PPO_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not runs:
            raise FileNotFoundError("No PPO runs found in ~/ray_results")
        
        print(f"🔍 Auto-detected: {runs[0].name}")
        return runs[0]
    
    def _find_progress_csv(self):
        """Find progress.csv in trial subdirectory"""
        csv_files = list(self.results_dir.glob("**/progress.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No progress.csv found in {self.results_dir}")
        
        print(f"📊 Using CSV: {csv_files[0]}")
        return csv_files[0]
    
    def _load_config(self):
        """Try to load training config"""
        config_files = list(self.results_dir.glob("**/params.json"))
        if config_files:
            try:
                with open(config_files[0]) as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def load_data(self):
        """Load training data from progress.csv"""
        print("📖 Loading training data...")
        df = pd.read_csv(self.csv_path)
        
        # Detect available columns
        self.available_metrics = {
            'won_objective': 'custom_metrics/won_objective_mean' in df.columns,
            'num_guests': 'custom_metrics/num_guests_mean' in df.columns,
            'park_rating': 'custom_metrics/park_rating_mean' in df.columns,
            'park_value': 'custom_metrics/park_value_mean' in df.columns,
            'company_value': 'custom_metrics/company_value_mean' in df.columns,
            'cash': 'custom_metrics/cash_mean' in df.columns,
            'loan': 'custom_metrics/loan_mean' in df.columns,
            'guests_in_park': 'custom_metrics/guests_in_park_mean' in df.columns,
            'admissions': 'custom_metrics/admissions_mean' in df.columns,
            'gpu': 'perf/gpu_util_percent0' in df.columns,
            'policy_loss': 'info/learner/default_policy/learner_stats/policy_loss' in df.columns,
            'vf_loss': 'info/learner/default_policy/learner_stats/vf_loss' in df.columns,
            'entropy': 'info/learner/default_policy/learner_stats/entropy' in df.columns,
            'kl': 'info/learner/default_policy/learner_stats/kl' in df.columns,
        }
        
        print(f"✅ Loaded {len(df)} iterations")
        print(f"   Available metrics: {', '.join([k for k, v in self.available_metrics.items() if v])}")
        
        return df
    
    def compute_comprehensive_statistics(self, df):
        """Compute comprehensive training statistics"""
        print("\n📈 Computing comprehensive statistics...")
        
        stats = {
            '=== TRAINING OVERVIEW ===': '',
            'Total Iterations': len(df),
            'Total Episodes': int(df['episodes_total'].iloc[-1]) if 'episodes_total' in df.columns else 0,
            'Total Timesteps': int(df['num_env_steps_sampled'].iloc[-1]) if 'num_env_steps_sampled' in df.columns else 0,
            'Training Time (hours)': df['time_total_s'].iloc[-1] / 3600 if 'time_total_s' in df.columns else 0,
            'Timesteps per Second': int(df['num_env_steps_sampled'].iloc[-1] / df['time_total_s'].iloc[-1]) if 'time_total_s' in df.columns else 0,
        }
        
        # === REWARD STATISTICS ===
        stats['=== REWARD ANALYSIS ==='] = ''
        stats['Initial Reward'] = df['episode_reward_mean'].iloc[0]
        stats['Final Reward'] = df['episode_reward_mean'].iloc[-1]
        stats['Best Reward'] = df['episode_reward_mean'].max()
        stats['Worst Reward'] = df['episode_reward_mean'].min()
        stats['Mean Reward'] = df['episode_reward_mean'].mean()
        stats['Median Reward'] = df['episode_reward_mean'].median()
        stats['Reward StdDev'] = df['episode_reward_mean'].std()
        stats['Reward Variance'] = df['episode_reward_mean'].var()
        stats['Reward Range'] = df['episode_reward_mean'].max() - df['episode_reward_mean'].min()
        stats['Absolute Change'] = df['episode_reward_mean'].iloc[-1] - df['episode_reward_mean'].iloc[0]
        
        if df['episode_reward_mean'].iloc[0] != 0:
            stats['Percent Change'] = ((df['episode_reward_mean'].iloc[-1] / df['episode_reward_mean'].iloc[0]) - 1) * 100
        else:
            stats['Percent Change'] = float('inf') if df['episode_reward_mean'].iloc[-1] > 0 else 0
        
        # Compute trend (linear regression)
        x = np.arange(len(df))
        y = df['episode_reward_mean'].values
        z = np.polyfit(x, y, 1)
        stats['Reward Trend (slope)'] = z[0]
        
        # === EPISODE LENGTH ===
        if 'episode_len_mean' in df.columns:
            stats['=== EPISODE LENGTH ==='] = ''
            stats['Avg Episode Length'] = df['episode_len_mean'].mean()
            stats['Max Episode Length'] = df['episode_len_mean'].max()
            stats['Min Episode Length'] = df['episode_len_mean'].min()
        
        # === SUCCESS METRICS ===
        if self.available_metrics['won_objective']:
            stats['=== SUCCESS METRICS ==='] = ''
            final_won = df['custom_metrics/won_objective_mean'].iloc[-1]
            initial_won = df['custom_metrics/won_objective_mean'].iloc[0]
            stats['Initial Win Rate (%)'] = (1 + initial_won) * 50
            stats['Final Win Rate (%)'] = (1 + final_won) * 50
            stats['Best Win Rate (%)'] = (1 + df['custom_metrics/won_objective_mean'].max()) * 50
            stats['Mean Win Rate (%)'] = (1 + df['custom_metrics/won_objective_mean'].mean()) * 50
            stats['Win Rate Improvement (%)'] = stats['Final Win Rate (%)'] - stats['Initial Win Rate (%)']
        
        # === PARK METRICS ===
        park_metrics = ['num_guests', 'park_rating', 'park_value', 'company_value', 
                       'cash', 'guests_in_park', 'admissions']
        has_park_metrics = any(self.available_metrics.get(m, False) for m in park_metrics)
        
        if has_park_metrics:
            stats['=== PARK PERFORMANCE ==='] = ''
            
            if self.available_metrics['num_guests']:
                col = 'custom_metrics/num_guests_mean'
                stats['Initial Avg Guests'] = df[col].iloc[0]
                stats['Final Avg Guests'] = df[col].iloc[-1]
                stats['Peak Avg Guests'] = df[col].max()
                stats['Guests Growth (%)'] = ((df[col].iloc[-1] / df[col].iloc[0]) - 1) * 100 if df[col].iloc[0] != 0 else 0
            
            if self.available_metrics['park_rating']:
                col = 'custom_metrics/park_rating_mean'
                stats['Initial Park Rating'] = df[col].iloc[0]
                stats['Final Park Rating'] = df[col].iloc[-1]
                stats['Best Park Rating'] = df[col].max()
            
            if self.available_metrics['park_value']:
                col = 'custom_metrics/park_value_mean'
                stats['Final Park Value'] = df[col].iloc[-1]
                stats['Peak Park Value'] = df[col].max()
            
            if self.available_metrics['company_value']:
                col = 'custom_metrics/company_value_mean'
                stats['Final Company Value'] = df[col].iloc[-1]
                stats['Peak Company Value'] = df[col].max()
            
            if self.available_metrics['cash']:
                col = 'custom_metrics/cash_mean'
                stats['Final Cash'] = df[col].iloc[-1]
                stats['Peak Cash'] = df[col].max()
            
            if self.available_metrics['admissions']:
                col = 'custom_metrics/admissions_mean'
                stats['Total Admissions'] = df[col].sum()
                stats['Avg Admissions per Episode'] = df[col].mean()
        
        # === LEARNING METRICS ===
        learning_metrics = ['policy_loss', 'vf_loss', 'entropy', 'kl']
        has_learning = any(self.available_metrics.get(m, False) for m in learning_metrics)
        
        if has_learning:
            stats['=== LEARNING DYNAMICS ==='] = ''
            
            if self.available_metrics['policy_loss']:
                col = 'info/learner/default_policy/learner_stats/policy_loss'
                stats['Final Policy Loss'] = df[col].iloc[-1]
                stats['Mean Policy Loss'] = df[col].mean()
                stats['Policy Loss StdDev'] = df[col].std()
            
            if self.available_metrics['vf_loss']:
                col = 'info/learner/default_policy/learner_stats/vf_loss'
                stats['Final Value Loss'] = df[col].iloc[-1]
                stats['Mean Value Loss'] = df[col].mean()
                stats['Value Loss StdDev'] = df[col].std()
            
            if self.available_metrics['entropy']:
                col = 'info/learner/default_policy/learner_stats/entropy'
                stats['Final Entropy'] = df[col].iloc[-1]
                stats['Mean Entropy'] = df[col].mean()
                stats['Initial Entropy'] = df[col].iloc[0]
                stats['Entropy Change (%)'] = ((df[col].iloc[-1] / df[col].iloc[0]) - 1) * 100 if df[col].iloc[0] != 0 else 0
            
            if self.available_metrics['kl']:
                col = 'info/learner/default_policy/learner_stats/kl'
                stats['Final KL Divergence'] = df[col].iloc[-1]
                stats['Mean KL Divergence'] = df[col].mean()
                stats['Max KL Divergence'] = df[col].max()
        
        # === PERFORMANCE METRICS ===
        stats['=== PERFORMANCE ==='] = ''
        if 'time_total_s' in df.columns and len(df) > 1:
            time_per_iter = (df['time_total_s'].iloc[-1] - df['time_total_s'].iloc[0]) / (len(df) - 1)
            stats['Avg Time per Iteration (s)'] = time_per_iter
            stats['Iterations per Hour'] = 3600 / time_per_iter if time_per_iter > 0 else 0
        
        if self.available_metrics['gpu']:
            stats['Avg GPU Utilization (%)'] = df['perf/gpu_util_percent0'].mean()
            stats['Peak GPU Utilization (%)'] = df['perf/gpu_util_percent0'].max()
            
            if 'perf/vram_util_percent0' in df.columns:
                stats['Avg VRAM Utilization (%)'] = df['perf/vram_util_percent0'].mean()
                stats['Peak VRAM Utilization (%)'] = df['perf/vram_util_percent0'].max()
        
        # === STABILITY METRICS ===
        stats['=== TRAINING STABILITY ==='] = ''
        
        # Coefficient of variation
        cv = (df['episode_reward_mean'].std() / abs(df['episode_reward_mean'].mean())) * 100 if df['episode_reward_mean'].mean() != 0 else 0
        stats['Reward Coefficient of Variation (%)'] = cv
        
        # Check for training instability (large sudden drops)
        reward_diff = df['episode_reward_mean'].diff()
        large_drops = (reward_diff < -abs(df['episode_reward_mean'].std())).sum()
        stats['Number of Large Drops'] = large_drops
        
        # Moving average convergence
        window = min(20, len(df) // 4)
        if window > 1:
            ma = df['episode_reward_mean'].rolling(window=window).mean()
            ma_std = ma.std()
            stats[f'MA{window} StdDev'] = ma_std
            stats['Convergence Score'] = 1.0 / (1.0 + ma_std) if ma_std > 0 else 1.0
        
        return stats
    
    def generate_comprehensive_plots(self, df, stats):
        """Generate comprehensive analysis plots"""
        print("\n📊 Generating comprehensive plots...")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # === ROW 1: REWARD ANALYSIS ===
        
        # 1. Reward progression with confidence bands
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(df['training_iteration'], df['episode_reward_mean'], 
                linewidth=2, label='Mean Reward', color='#2E86DE')
        
        if 'episode_reward_max' in df.columns and 'episode_reward_min' in df.columns:
            ax1.fill_between(df['training_iteration'], 
                           df['episode_reward_min'], 
                           df['episode_reward_max'],
                           alpha=0.2, label='Min-Max Range', color='#2E86DE')
        
        # Add trend line
        z = np.polyfit(df['training_iteration'], df['episode_reward_mean'], 1)
        p = np.poly1d(z)
        ax1.plot(df['training_iteration'], p(df['training_iteration']), 
                "--", linewidth=2, alpha=0.7, label='Trend', color='#EE5A6F')
        
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
        ax1.set_title('Episode Reward Progression', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Smoothed reward with multiple windows
        ax2 = fig.add_subplot(gs[0, 2:4])
        windows = [5, 20, 50]
        colors = ['#F79F1F', '#EA2027', '#0652DD']
        
        for window, color in zip(windows, colors):
            if window < len(df):
                smoothed = df['episode_reward_mean'].rolling(window=window, min_periods=1).mean()
                ax2.plot(df['training_iteration'], smoothed, 
                        linewidth=2, label=f'MA{window}', color=color)
        
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Smoothed Reward', fontsize=11, fontweight='bold')
        ax2.set_title('Reward Trends (Multiple Moving Averages)', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # === ROW 2: DISTRIBUTIONS AND WIN RATE ===
        
        # 3. Reward distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(df['episode_reward_mean'], bins=min(40, len(df)//3), 
                edgecolor='black', alpha=0.7, color='#5F27CD')
        ax3.axvline(df['episode_reward_mean'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {df["episode_reward_mean"].mean():.2f}')
        ax3.axvline(df['episode_reward_mean'].median(), color='orange', 
                   linestyle='--', linewidth=2, label=f'Median: {df["episode_reward_mean"].median():.2f}')
        ax3.set_xlabel('Reward', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax3.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Reward boxplot over time (quartiles)
        ax4 = fig.add_subplot(gs[1, 1])
        n_boxes = min(10, len(df) // 10)
        if n_boxes > 1:
            box_data = []
            box_positions = []
            chunk_size = len(df) // n_boxes
            
            for i in range(n_boxes):
                start = i * chunk_size
                end = start + chunk_size if i < n_boxes - 1 else len(df)
                chunk = df['episode_reward_mean'].iloc[start:end]
                box_data.append(chunk.values)
                box_positions.append(df['training_iteration'].iloc[start + chunk_size // 2])
            
            bp = ax4.boxplot(box_data, positions=box_positions, widths=box_positions[1]-box_positions[0] if len(box_positions) > 1 else 10)
            ax4.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Reward', fontsize=10, fontweight='bold')
            ax4.set_title('Reward Distribution Over Time', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Win rate (if available)
        ax5 = fig.add_subplot(gs[1, 2])
        if self.available_metrics['won_objective']:
            win_rate = (1 + df['custom_metrics/won_objective_mean']) * 50
            ax5.plot(df['training_iteration'], win_rate, 
                    color='#27AE60', linewidth=2.5, label='Win Rate')
            
            # Add smoothed version
            if len(df) > 10:
                win_smooth = win_rate.rolling(window=10, min_periods=1).mean()
                ax5.plot(df['training_iteration'], win_smooth, 
                        color='#F39C12', linewidth=2, linestyle='--', label='Smoothed', alpha=0.8)
            
            ax5.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (random)')
            ax5.fill_between(df['training_iteration'], 50, win_rate, 
                           where=(win_rate >= 50), alpha=0.2, color='green', label='Above Random')
            ax5.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax5.set_ylabel('Win Rate (%)', fontsize=10, fontweight='bold')
            ax5.set_title('Win Rate Progression', fontsize=12, fontweight='bold')
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, 100])
        else:
            ax5.text(0.5, 0.5, 'Win Rate\nData Not Available', 
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.axis('off')
        
        # 6. Episode length (if available)
        ax6 = fig.add_subplot(gs[1, 3])
        if 'episode_len_mean' in df.columns:
            ax6.plot(df['training_iteration'], df['episode_len_mean'], 
                    color='#8E44AD', linewidth=2)
            ax6.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax6.set_ylabel('Episode Length', fontsize=10, fontweight='bold')
            ax6.set_title('Episode Length Over Time', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Episode Length\nData Not Available', 
                    ha='center', va='center', fontsize=12, transform=ax6.transAxes)
            ax6.axis('off')
        
        # === ROW 3: PARK METRICS ===
        
        # 7. Number of guests
        ax7 = fig.add_subplot(gs[2, 0])
        if self.available_metrics['num_guests']:
            col = 'custom_metrics/num_guests_mean'
            ax7.plot(df['training_iteration'], df[col], 
                    color='#16A085', linewidth=2)
            ax7.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax7.set_ylabel('Number of Guests', fontsize=10, fontweight='bold')
            ax7.set_title('Average Guests Progression', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Guests Data\nNot Available', 
                    ha='center', va='center', fontsize=12, transform=ax7.transAxes)
            ax7.axis('off')
        
        # 8. Park rating
        ax8 = fig.add_subplot(gs[2, 1])
        if self.available_metrics['park_rating']:
            col = 'custom_metrics/park_rating_mean'
            ax8.plot(df['training_iteration'], df[col], 
                    color='#E67E22', linewidth=2)
            ax8.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax8.set_ylabel('Park Rating', fontsize=10, fontweight='bold')
            ax8.set_title('Park Rating Progression', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Park Rating\nData Not Available', 
                    ha='center', va='center', fontsize=12, transform=ax8.transAxes)
            ax8.axis('off')
        
        # 9. Park/Company value
        ax9 = fig.add_subplot(gs[2, 2])
        has_value_data = False
        if self.available_metrics['park_value']:
            col = 'custom_metrics/park_value_mean'
            ax9.plot(df['training_iteration'], df[col], 
                    color='#3498DB', linewidth=2, label='Park Value')
            has_value_data = True
        
        if self.available_metrics['company_value']:
            col = 'custom_metrics/company_value_mean'
            ax9.plot(df['training_iteration'], df[col], 
                    color='#9B59B6', linewidth=2, label='Company Value')
            has_value_data = True
        
        if has_value_data:
            ax9.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax9.set_ylabel('Value', fontsize=10, fontweight='bold')
            ax9.set_title('Park/Company Value', fontsize=12, fontweight='bold')
            ax9.legend(loc='best')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Value Data\nNot Available', 
                    ha='center', va='center', fontsize=12, transform=ax9.transAxes)
            ax9.axis('off')
        
        # 10. Cash/Finances
        ax10 = fig.add_subplot(gs[2, 3])
        has_finance_data = False
        if self.available_metrics['cash']:
            col = 'custom_metrics/cash_mean'
            ax10.plot(df['training_iteration'], df[col], 
                     color='#27AE60', linewidth=2, label='Cash')
            has_finance_data = True
        
        if self.available_metrics['loan']:
            col = 'custom_metrics/loan_mean'
            ax10.plot(df['training_iteration'], df[col], 
                     color='#E74C3C', linewidth=2, label='Loan')
            has_finance_data = True
        
        if has_finance_data:
            ax10.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax10.set_ylabel('Amount', fontsize=10, fontweight='bold')
            ax10.set_title('Financial Metrics', fontsize=12, fontweight='bold')
            ax10.legend(loc='best')
            ax10.grid(True, alpha=0.3)
        else:
            ax10.text(0.5, 0.5, 'Financial Data\nNot Available', 
                     ha='center', va='center', fontsize=12, transform=ax10.transAxes)
            ax10.axis('off')
        
        # === ROW 4: LEARNING DYNAMICS AND SUMMARY ===
        
        # 11. Policy and Value Loss
        ax11 = fig.add_subplot(gs[3, 0])
        has_loss_data = False
        if self.available_metrics['policy_loss']:
            col = 'info/learner/default_policy/learner_stats/policy_loss'
            ax11.plot(df['training_iteration'], df[col], 
                     color='#E74C3C', linewidth=2, label='Policy Loss')
            has_loss_data = True
        
        if self.available_metrics['vf_loss']:
            col = 'info/learner/default_policy/learner_stats/vf_loss'
            ax11_twin = ax11.twinx() if has_loss_data else ax11
            ax11_twin.plot(df['training_iteration'], df[col], 
                          color='#3498DB', linewidth=2, label='Value Loss')
            ax11_twin.set_ylabel('Value Loss', fontsize=10, fontweight='bold')
            has_loss_data = True
        
        if has_loss_data:
            ax11.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax11.set_ylabel('Policy Loss', fontsize=10, fontweight='bold')
            ax11.set_title('Training Losses', fontsize=12, fontweight='bold')
            ax11.legend(loc='upper left')
            if self.available_metrics['vf_loss']:
                ax11_twin.legend(loc='upper right')
            ax11.grid(True, alpha=0.3)
        else:
            ax11.text(0.5, 0.5, 'Loss Data\nNot Available', 
                     ha='center', va='center', fontsize=12, transform=ax11.transAxes)
            ax11.axis('off')
        
        # 12. Entropy and KL Divergence
        ax12 = fig.add_subplot(gs[3, 1])
        has_explore_data = False
        if self.available_metrics['entropy']:
            col = 'info/learner/default_policy/learner_stats/entropy'
            ax12.plot(df['training_iteration'], df[col], 
                     color='#F39C12', linewidth=2, label='Entropy')
            has_explore_data = True
        
        if self.available_metrics['kl']:
            col = 'info/learner/default_policy/learner_stats/kl'
            ax12_twin = ax12.twinx() if has_explore_data else ax12
            ax12_twin.plot(df['training_iteration'], df[col], 
                          color='#9B59B6', linewidth=2, label='KL Divergence')
            ax12_twin.set_ylabel('KL Divergence', fontsize=10, fontweight='bold')
            has_explore_data = True
        
        if has_explore_data:
            ax12.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax12.set_ylabel('Entropy', fontsize=10, fontweight='bold')
            ax12.set_title('Exploration Metrics', fontsize=12, fontweight='bold')
            ax12.legend(loc='upper left')
            if self.available_metrics['kl']:
                ax12_twin.legend(loc='upper right')
            ax12.grid(True, alpha=0.3)
        else:
            ax12.text(0.5, 0.5, 'Exploration Data\nNot Available', 
                     ha='center', va='center', fontsize=12, transform=ax12.transAxes)
            ax12.axis('off')
        
        # 13. Hardware utilization
        ax13 = fig.add_subplot(gs[3, 2])
        if self.available_metrics['gpu']:
            ax13.plot(df['training_iteration'], df['perf/gpu_util_percent0'], 
                     color='#16A085', linewidth=2, label='GPU %')
            if 'perf/vram_util_percent0' in df.columns:
                ax13.plot(df['training_iteration'], df['perf/vram_util_percent0'], 
                         color='#E67E22', linewidth=2, label='VRAM %')
            ax13.set_xlabel('Iteration', fontsize=10, fontweight='bold')
            ax13.set_ylabel('Utilization (%)', fontsize=10, fontweight='bold')
            ax13.set_title('Hardware Usage', fontsize=12, fontweight='bold')
            ax13.legend(loc='best')
            ax13.grid(True, alpha=0.3)
            ax13.set_ylim([0, 100])
        else:
            ax13.text(0.5, 0.5, 'GPU Data\nNot Available', 
                     ha='center', va='center', fontsize=12, transform=ax13.transAxes)
            ax13.axis('off')
        
        # 14. Summary text
        ax14 = fig.add_subplot(gs[3, 3])
        ax14.axis('off')
        
        summary_lines = [
            "📊 TRAINING SUMMARY",
            "=" * 35,
            f"Iterations: {stats['Total Iterations']:,}",
            f"Episodes: {stats['Total Episodes']:,}",
            f"Timesteps: {stats['Total Timesteps']:,}",
            f"Time: {stats['Training Time (hours)']:.1f}h",
            "",
            "🎯 REWARD",
            f"Final: {stats['Final Reward']:.4f}",
            f"Best: {stats['Best Reward']:.4f}",
            f"Change: {stats['Absolute Change']:+.4f}",
            f"Trend: {stats['Reward Trend (slope)']:+.6f}",
            ""
        ]
        
        if 'Final Win Rate (%)' in stats:
            summary_lines.extend([
                "🏆 SUCCESS",
                f"Win Rate: {stats['Final Win Rate (%)']:.1f}%",
                f"Improvement: {stats['Win Rate Improvement (%)']:+.1f}%",
                ""
            ])
        
        if 'Final Avg Guests' in stats:
            summary_lines.extend([
                "👥 PARK",
                f"Guests: {stats['Final Avg Guests']:.0f}",
            ])
            if 'Final Park Rating' in stats:
                summary_lines.append(f"Rating: {stats['Final Park Rating']:.1f}")
            summary_lines.append("")
        
        if 'Convergence Score' in stats:
            summary_lines.extend([
                "📈 STABILITY",
                f"Score: {stats['Convergence Score']:.3f}",
                f"CV: {stats['Reward Coefficient of Variation (%)']:.1f}%",
            ])
        
        summary_text = "\n".join(summary_lines)
        ax14.text(0.05, 0.95, summary_text, fontsize=9.5, family='monospace',
                 verticalalignment='top', transform=ax14.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Overall title
        fig.suptitle(f'RCT-RL Training Analysis - {self.results_dir.name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        plot_file = self.output_dir / f"comprehensive_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Comprehensive plots saved: {plot_file}")
        
        return plot_file
    
    def generate_detailed_report(self, df, stats):
        """Generate detailed HTML report"""
        print("\n📄 Generating HTML report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        report_file = self.output_dir / f"detailed_report_{timestamp}.html"
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RCT-RL Training Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .stat-change {{
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tr:hover {{
            background: #e9ecef;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 20px;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎢 RCT-RL Training Analysis Report</h1>
            <p>{self.results_dir.name}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
"""
        
        # === KEY METRICS ===
        html_content += """
            <div class="section">
                <h2>📊 Key Performance Indicators</h2>
                <div class="stats-grid">
"""
        
        # Reward card
        reward_change = stats.get('Percent Change', 0)
        reward_class = 'positive' if reward_change > 0 else 'negative'
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">Final Reward</div>
                        <div class="stat-value">{stats['Final Reward']:.4f}</div>
                        <div class="stat-change {reward_class}">
                            {reward_change:+.1f}% from initial
                        </div>
                    </div>
"""
        
        # Win rate card (if available)
        if 'Final Win Rate (%)' in stats:
            win_rate = stats['Final Win Rate (%)']
            badge = 'success' if win_rate >= 70 else 'warning' if win_rate >= 50 else 'danger'
            html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value">{win_rate:.1f}%
                            <span class="badge badge-{badge}">
                                {'Excellent' if win_rate >= 70 else 'Good' if win_rate >= 50 else 'Needs Work'}
                            </span>
                        </div>
                        <div class="stat-change">
                            {stats['Win Rate Improvement (%)']:+.1f}% improvement
                        </div>
                    </div>
"""
        
        # Training time card
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">Training Duration</div>
                        <div class="stat-value">{stats['Training Time (hours)']:.1f}h</div>
                        <div class="stat-change">
                            {stats['Total Timesteps']:,} timesteps
                        </div>
                    </div>
"""
        
        # Iterations card
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">Total Iterations</div>
                        <div class="stat-value">{stats['Total Iterations']:,}</div>
                        <div class="stat-change">
                            {stats['Total Episodes']:,} episodes
                        </div>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
"""
        
        # === DETAILED STATISTICS TABLE ===
        html_content += """
            <div class="section">
                <h2>📈 Detailed Statistics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for key, value in stats.items():
            if key.startswith('==='):
                html_content += f"""
                        <tr style="background: #667eea; color: white;">
                            <td colspan="2" style="font-weight: bold; font-size: 1.1em;">{value if value else key.replace('===', '').strip()}</td>
                        </tr>
"""
            else:
                if isinstance(value, (int, np.integer)):
                    value_str = f"{value:,}"
                elif isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                html_content += f"""
                        <tr>
                            <td>{key}</td>
                            <td><strong>{value_str}</strong></td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
"""
        
        # === TRAINING INSIGHTS ===
        html_content += """
            <div class="section">
                <h2>💡 Training Insights</h2>
"""
        
        # Generate insights
        insights = []
        
        # Reward trend
        if stats.get('Reward Trend (slope)', 0) > 0:
            insights.append("✅ <strong>Positive Learning Trend:</strong> The agent is consistently improving over time.")
        else:
            insights.append("⚠️ <strong>Stagnant or Declining:</strong> The reward trend is not increasing. Consider adjusting hyperparameters.")
        
        # Win rate
        if 'Final Win Rate (%)' in stats:
            if stats['Final Win Rate (%)'] >= 70:
                insights.append("✅ <strong>Excellent Success Rate:</strong> The agent has learned to win consistently!")
            elif stats['Final Win Rate (%)'] >= 50:
                insights.append("🎯 <strong>Good Progress:</strong> Win rate is above random chance. Further training may improve performance.")
            else:
                insights.append("⚠️ <strong>Low Win Rate:</strong> The agent is not performing better than random. Review reward structure and environment.")
        
        # Stability
        cv = stats.get('Reward Coefficient of Variation (%)', 0)
        if cv < 20:
            insights.append("✅ <strong>Stable Training:</strong> Low reward variance indicates consistent performance.")
        elif cv < 50:
            insights.append("🎯 <strong>Moderate Stability:</strong> Some variance in performance is normal during learning.")
        else:
            insights.append("⚠️ <strong>High Variance:</strong> Large reward fluctuations detected. Consider tuning exploration parameters.")
        
        # Convergence
        if 'Convergence Score' in stats:
            conv = stats['Convergence Score']
            if conv > 0.8:
                insights.append("✅ <strong>Near Convergence:</strong> The policy appears to have stabilized.")
            elif conv > 0.5:
                insights.append("🎯 <strong>Approaching Convergence:</strong> The policy is stabilizing but may benefit from more training.")
            else:
                insights.append("⚠️ <strong>Still Learning:</strong> The policy has not converged yet. More training iterations recommended.")
        
        for insight in insights:
            html_content += f"""
                <div class="highlight">
                    {insight}
                </div>
"""
        
        html_content += """
            </div>
"""
        
        # === CONFIGURATION ===
        if self.config:
            html_content += """
            <div class="section">
                <h2>⚙️ Training Configuration</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            
            # Extract relevant config
            if 'env_config' in self.config:
                for key, value in self.config['env_config'].items():
                    html_content += f"""
                        <tr>
                            <td>env_{key}</td>
                            <td>{value}</td>
                        </tr>
"""
            
            # PPO params
            ppo_params = ['lr', 'gamma', 'lambda', 'clip_param', 'vf_clip_param', 
                         'entropy_coeff', 'train_batch_size', 'sgd_minibatch_size']
            for param in ppo_params:
                if param in self.config:
                    html_content += f"""
                        <tr>
                            <td>{param}</td>
                            <td>{self.config[param]}</td>
                        </tr>
"""
            
            html_content += """
                    </tbody>
                </table>
            </div>
"""
        
        # === FOOTER ===
        html_content += f"""
        </div>
        
        <div class="footer">
            <p>Report generated by Enhanced RCT-RL Analyzer</p>
            <p>Training run: {self.results_dir.name}</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"💾 HTML report saved: {report_file}")
        
        return report_file
    
    def save_comprehensive_data(self, df, stats):
        """Export comprehensive data"""
        print("\n💾 Saving comprehensive data...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # === 1. Key metrics CSV ===
        key_cols = ['training_iteration', 'episode_reward_mean', 'episode_reward_max', 
                    'episode_reward_min', 'episodes_total', 'num_env_steps_sampled']
        
        # Add all available custom metrics
        for col in df.columns:
            if 'custom_metrics' in col or 'learner_stats' in col:
                key_cols.append(col)
        
        available_cols = [c for c in key_cols if c in df.columns]
        key_df = df[available_cols].copy()
        
        # Add computed columns
        if self.available_metrics['won_objective']:
            key_df['win_rate_percent'] = (1 + df['custom_metrics/won_objective_mean']) * 50
        
        # Add smoothed reward
        key_df['reward_ma5'] = df['episode_reward_mean'].rolling(window=5, min_periods=1).mean()
        key_df['reward_ma20'] = df['episode_reward_mean'].rolling(window=20, min_periods=1).mean()
        
        csv_file = self.output_dir / f"comprehensive_metrics_{timestamp}.csv"
        key_df.to_csv(csv_file, index=False)
        print(f"   📊 Metrics CSV: {csv_file}")
        
        # === 2. Statistics JSON ===
        json_file = self.output_dir / f"statistics_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_stats = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.floating)):
                json_stats[key] = float(value)
            elif isinstance(value, (int, float, str)):
                json_stats[key] = value
            else:
                json_stats[key] = str(value)
        
        with open(json_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        print(f"   📊 Statistics JSON: {json_file}")
        
        # === 3. Statistics TXT ===
        stats_file = self.output_dir / f"statistics_{timestamp}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RCT-RL TRAINING STATISTICS - COMPREHENSIVE REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Run: {self.results_dir.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV Path: {self.csv_path}\n")
            f.write("="*70 + "\n\n")
            
            for key, value in stats.items():
                if key.startswith('==='):
                    f.write(f"\n{key}\n")
                    f.write("-"*70 + "\n")
                else:
                    if isinstance(value, (int, np.integer)):
                        f.write(f"{key:.<55} {value:>14,}\n")
                    elif isinstance(value, (float, np.floating)):
                        f.write(f"{key:.<55} {value:>14.6f}\n")
                    else:
                        f.write(f"{key:.<55} {str(value):>14}\n")
        
        print(f"   📄 Statistics TXT: {stats_file}")
        
        # === 4. Summary README ===
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# RCT-RL Training Analysis\n\n")
            f.write(f"**Run:** `{self.results_dir.name}`\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Quick Summary\n\n")
            f.write(f"- **Total Iterations:** {stats['Total Iterations']:,}\n")
            f.write(f"- **Total Episodes:** {stats['Total Episodes']:,}\n")
            f.write(f"- **Training Time:** {stats['Training Time (hours)']:.1f} hours\n")
            f.write(f"- **Final Reward:** {stats['Final Reward']:.4f}\n")
            
            if 'Final Win Rate (%)' in stats:
                f.write(f"- **Win Rate:** {stats['Final Win Rate (%)']:.1f}%\n")
            
            f.write("\n## Files Generated\n\n")
            f.write(f"- `comprehensive_analysis_{timestamp}.png` - Visualization dashboard\n")
            f.write(f"- `comprehensive_metrics_{timestamp}.csv` - Time series data\n")
            f.write(f"- `statistics_{timestamp}.json` - All statistics in JSON format\n")
            f.write(f"- `statistics_{timestamp}.txt` - Human-readable statistics\n")
            f.write(f"- `detailed_report_{timestamp}.html` - Interactive HTML report\n")
            f.write("\n## Key Findings\n\n")
            
            if stats.get('Reward Trend (slope)', 0) > 0:
                f.write("✅ Positive learning trend detected\n")
            
            if 'Final Win Rate (%)' in stats and stats['Final Win Rate (%)'] >= 50:
                f.write("✅ Agent performing above random baseline\n")
            
            cv = stats.get('Reward Coefficient of Variation (%)', 0)
            if cv < 30:
                f.write("✅ Stable training performance\n")
        
        print(f"   📖 README: {readme_file}")
        
        return csv_file, json_file, stats_file, readme_file
    
    def run(self):
        """Run complete comprehensive analysis"""
        print("\n" + "="*70)
        print("🚀 STARTING ENHANCED RCT-RL TRAINING ANALYSIS")
        print("="*70)
        
        # Load data
        df = self.load_data()
        
        # Compute statistics
        stats = self.compute_comprehensive_statistics(df)
        
        # Generate visualizations
        plot_file = self.generate_comprehensive_plots(df, stats)
        
        # Generate HTML report
        report_file = self.generate_detailed_report(df, stats)
        
        # Save data
        csv_file, json_file, stats_file, readme_file = self.save_comprehensive_data(df, stats)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n📁 Output directory: {self.output_dir}/")
        print("\n📊 Generated files:")
        print(f"   1. {plot_file.name} (comprehensive visualization)")
        print(f"   2. {report_file.name} (interactive HTML report)")
        print(f"   3. {csv_file.name} (time series data)")
        print(f"   4. {json_file.name} (statistics JSON)")
        print(f"   5. {stats_file.name} (statistics text)")
        print(f"   6. {readme_file.name} (summary)")
        
        print("\n🎯 Quick Stats:")
        print(f"   Reward: {stats['Initial Reward']:.4f} → {stats['Final Reward']:.4f} ({stats.get('Percent Change', 0):+.1f}%)")
        if 'Final Win Rate (%)' in stats:
            print(f"   Win Rate: {stats['Initial Win Rate (%)']:.1f}% → {stats['Final Win Rate (%)']:.1f}%")
        print(f"   Training: {stats['Total Iterations']} iterations, {stats['Training Time (hours)']:.1f}h")
        print()
        
        return df, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced RCT-RL Training Analyzer - Comprehensive analysis and reporting"
    )
    parser.add_argument('--run', type=str, 
                       help='Path to PPO run directory (auto-detect latest if omitted)')
    parser.add_argument('--output', type=str, 
                       help='Output directory (default: run_dir/analysis)')
    
    args = parser.parse_args()
    
    try:
        analyzer = EnhancedRCTRLAnalyzer(results_dir=args.run, output_dir=args.output)
        analyzer.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
