#!/usr/bin/env python3
"""
Generate visualizations for the evaluation results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ResultsVisualizer:
    """Creates visualizations for evaluation results."""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.charts_dir = self.results_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_data(self):
        """Load aggregated results."""
        with open(self.results_dir / "aggregated_results.json", 'r') as f:
            self.results = json.load(f)
        
        self.summary_df = pd.read_csv(self.results_dir / "summary.csv")
        self.domain_df = pd.read_csv(self.results_dir / "domain_breakdown.csv")
    
    def create_overall_comparison_chart(self):
        """Create overall performance comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GPT-5 vs Claude 4 Sonnet: Overall Performance Comparison', fontsize=16, fontweight='bold')
        
        # Metrics to plot
        metrics = [
            ('task_success', 'Task Success Rate'),
            ('factual_precision', 'Factual Precision'),
            ('reasoning_quality', 'Reasoning Quality'),
            ('helpfulness', 'Helpfulness')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Get data
            gpt5_val = self.summary_df[self.summary_df['model'] == 'gpt-5'][metric].iloc[0]
            claude4_val = self.summary_df[self.summary_df['model'] == 'claude-4-sonnet'][metric].iloc[0]
            
            # Create bar chart
            models = ['GPT-5', 'Claude 4 Sonnet']
            values = [gpt5_val, claude4_val]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax.bar(models, values, color=colors, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_ylim(0, 1.0 if metric != 'reasoning_quality' else 5.0)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "overall_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_domain_breakdown_chart(self):
        """Create domain-specific performance breakdown."""
        # Prepare data
        domains = self.domain_df['domain'].unique()
        metrics = ['task_success', 'factual_precision', 'reasoning_quality', 'helpfulness']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance by Domain: GPT-5 vs Claude 4 Sonnet', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data for this metric
            gpt5_values = []
            claude4_values = []
            domain_labels = []
            
            for domain in domains:
                domain_data = self.domain_df[self.domain_df['domain'] == domain]
                
                gpt5_row = domain_data[domain_data['model'] == 'gpt-5']
                claude4_row = domain_data[domain_data['model'] == 'claude-4-sonnet']
                
                if not gpt5_row.empty and not claude4_row.empty:
                    gpt5_values.append(gpt5_row[metric].iloc[0])
                    claude4_values.append(claude4_row[metric].iloc[0])
                    domain_labels.append(domain.replace('_', ' ').title())
            
            # Create grouped bar chart
            x = np.arange(len(domain_labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, gpt5_values, width, label='GPT-5', color='#FF6B6B', alpha=0.8)
            bars2 = ax.bar(x + width/2, claude4_values, width, label='Claude 4 Sonnet', color='#4ECDC4', alpha=0.8)
            
            ax.set_xlabel('Domain')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Domain')
            ax.set_xticks(x)
            ax.set_xticklabels(domain_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "domain_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_latency_comparison_chart(self):
        """Create latency comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Latency Comparison: GPT-5 vs Claude 4 Sonnet', fontsize=16, fontweight='bold')
        
        # Overall latency comparison
        models = ['GPT-5', 'Claude 4 Sonnet']
        
        # Get latency data from summary
        gpt5_latencies = [
            self.summary_df[self.summary_df['model'] == 'gpt-5']['latency_p50'].iloc[0],
            self.summary_df[self.summary_df['model'] == 'gpt-5']['latency_p90'].iloc[0],
            self.summary_df[self.summary_df['model'] == 'gpt-5']['latency_p95'].iloc[0]
        ]
        
        claude4_latencies = [
            self.summary_df[self.summary_df['model'] == 'claude-4-sonnet']['latency_p50'].iloc[0],
            self.summary_df[self.summary_df['model'] == 'claude-4-sonnet']['latency_p90'].iloc[0],
            self.summary_df[self.summary_df['model'] == 'claude-4-sonnet']['latency_p95'].iloc[0]
        ]
        
        percentiles = ['p50', 'p90', 'p95']
        x = np.arange(len(percentiles))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, gpt5_latencies, width, label='GPT-5', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, claude4_latencies, width, label='Claude 4 Sonnet', color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('Percentile')
        ax1.set_ylabel('Latency (seconds)')
        ax1.set_title('Overall Latency Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(percentiles)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # Domain-specific latency
        domains = self.domain_df['domain'].unique()
        domain_labels = [d.replace('_', ' ').title() for d in domains]
        
        gpt5_domain_latencies = []
        claude4_domain_latencies = []
        
        for domain in domains:
            domain_data = self.domain_df[self.domain_df['domain'] == domain]
            
            gpt5_row = domain_data[domain_data['model'] == 'gpt-5']
            claude4_row = domain_data[domain_data['model'] == 'claude-4-sonnet']
            
            gpt5_lat = gpt5_row['latency_p50'].iloc[0] if not gpt5_row.empty else 0
            claude4_lat = claude4_row['latency_p50'].iloc[0] if not claude4_row.empty else 0
            
            gpt5_domain_latencies.append(gpt5_lat)
            claude4_domain_latencies.append(claude4_lat)
        
        x2 = np.arange(len(domain_labels))
        bars3 = ax2.bar(x2 - width/2, gpt5_domain_latencies, width, label='GPT-5', color='#FF6B6B', alpha=0.8)
        bars4 = ax2.bar(x2 + width/2, claude4_domain_latencies, width, label='Claude 4 Sonnet', color='#4ECDC4', alpha=0.8)
        
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Median Latency (seconds)')
        ax2.set_title('Latency by Domain (p50)')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(domain_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "latency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_radar_chart(self):
        """Create radar chart comparing overall capabilities."""
        # Prepare data
        metrics = ['Task Success', 'Factual Precision', 'Reasoning Quality', 
                  'Helpfulness', 'Conciseness']
        
        gpt5_values = []
        claude4_values = []
        
        metric_mapping = {
            'Task Success': 'task_success',
            'Factual Precision': 'factual_precision', 
            'Reasoning Quality': 'reasoning_quality',
            'Helpfulness': 'helpfulness',
            'Conciseness': 'conciseness'
        }
        
        for metric in metrics:
            col_name = metric_mapping[metric]
            gpt5_val = self.summary_df[self.summary_df['model'] == 'gpt-5'][col_name].iloc[0]
            claude4_val = self.summary_df[self.summary_df['model'] == 'claude-4-sonnet'][col_name].iloc[0]
            
            # Normalize reasoning_quality to 0-1 scale
            if col_name == 'reasoning_quality':
                gpt5_val /= 5.0
                claude4_val /= 5.0
            
            gpt5_values.append(gpt5_val)
            claude4_values.append(claude4_val)
        
        # Create radar chart using plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=gpt5_values + [gpt5_values[0]],  # Close the polygon
            theta=metrics + [metrics[0]],
            fill='toself',
            name='GPT-5',
            line_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=claude4_values + [claude4_values[0]],  # Close the polygon
            theta=metrics + [metrics[0]],
            fill='toself',
            name='Claude 4 Sonnet',
            line_color='#4ECDC4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Overall Capability Comparison: GPT-5 vs Claude 4 Sonnet"
        )
        
        fig.write_html(self.charts_dir / "radar_comparison.html")
        fig.write_image(self.charts_dir / "radar_comparison.png", width=800, height=600)
    
    def create_statistical_significance_chart(self):
        """Create chart showing statistical significance of differences."""
        stats_data = self.results['statistical_tests']
        
        metrics = list(stats_data.keys())
        p_values = [stats_data[m]['p_value'] for m in metrics]
        effect_sizes = [abs(stats_data[m]['cohen_d']) for m in metrics]
        significant = [stats_data[m]['significant'] for m in metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Statistical Analysis: GPT-5 vs Claude 4 Sonnet', fontsize=16, fontweight='bold')
        
        # P-values chart
        colors = ['red' if sig else 'gray' for sig in significant]
        bars1 = ax1.bar(range(len(metrics)), p_values, color=colors, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance (P-values)')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Effect sizes chart
        bars2 = ax2.bar(range(len(metrics)), effect_sizes, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Effect Size (|Cohen\'s d|)')
        ax2.set_title('Effect Sizes')
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add effect size interpretation lines
        ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small (0.2)')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium (0.5)')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large (0.8)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "statistical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization charts."""
        print("Loading data...")
        self.load_data()
        
        print("Creating overall comparison chart...")
        self.create_overall_comparison_chart()
        
        print("Creating domain breakdown chart...")
        self.create_domain_breakdown_chart()
        
        print("Creating latency comparison chart...")
        self.create_latency_comparison_chart()
        
        print("Creating radar chart...")
        self.create_radar_chart()
        
        print("Creating statistical significance chart...")
        self.create_statistical_significance_chart()
        
        print(f"All visualizations saved to {self.charts_dir}")

if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.generate_all_visualizations()
