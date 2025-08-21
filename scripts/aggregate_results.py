#!/usr/bin/env python3
"""
Aggregate evaluation results and generate summary statistics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ResultsAggregator:
    """Aggregates and analyzes evaluation results."""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.prompts_dir = Path("prompts")
    
    def load_evaluation_scores(self) -> pd.DataFrame:
        """Load evaluation scores from JSON file."""
        with open("results/evaluation_scores.json", 'r') as f:
            scores_data = json.load(f)
        
        # Flatten the nested structure
        rows = []
        for item in scores_data:
            prompt_id = item["prompt_id"]
            
            # Add GPT-5 scores
            gpt5_row = {"prompt_id": prompt_id, "model": "gpt-5"}
            gpt5_row.update(item["gpt5"])
            rows.append(gpt5_row)
            
            # Add Claude 4 scores
            claude4_row = {"prompt_id": prompt_id, "model": "claude-4-sonnet"}
            claude4_row.update(item["claude4"])
            rows.append(claude4_row)
        
        return pd.DataFrame(rows)
    
    def load_latency_data(self) -> pd.DataFrame:
        """Load latency data from JSON file."""
        with open("results/latency_raw.json", 'r') as f:
            latency_data = json.load(f)
        
        # Convert to long format
        rows = []
        for item in latency_data:
            rows.append({
                "prompt_id": item["prompt_id"],
                "domain": item["domain"],
                "model": "gpt-5",
                "latency": item["gpt5_latency"],
                "error": item.get("gpt5_error")
            })
            rows.append({
                "prompt_id": item["prompt_id"],
                "domain": item["domain"],
                "model": "claude-4-sonnet",
                "latency": item["claude4_latency"],
                "error": item.get("claude4_error")
            })
        
        return pd.DataFrame(rows)
    
    def add_domain_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain information to dataframe."""
        domain_map = {}
        
        # Load domain info from prompt files
        for prompt_file in self.prompts_dir.glob("*.json"):
            if prompt_file.name == "summary.json":
                continue
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)
                domain_map[prompt_data["id"]] = prompt_data["domain"]
        
        df["domain"] = df["prompt_id"].map(domain_map)
        return df
    
    def calculate_bootstrap_ci(self, data: np.array, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - ci
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def generate_overall_summary(self, scores_df: pd.DataFrame, latency_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall performance summary."""
        summary = {}
        
        # Overall metrics by model
        for model in ["gpt-5", "claude-4-sonnet"]:
            model_scores = scores_df[scores_df["model"] == model]
            model_latency = latency_df[latency_df["model"] == model]
            
            # Calculate means and confidence intervals
            metrics = {}
            for metric in ["task_success", "factual_precision", "reasoning_quality", 
                          "helpfulness", "conciseness", "hallucination_rate", 
                          "safety_refusal_correctness"]:
                values = model_scores[metric].values
                mean_val = np.mean(values)
                ci_lower, ci_upper = self.calculate_bootstrap_ci(values)
                
                metrics[metric] = {
                    "mean": float(mean_val),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper)
                }
            
            # Latency percentiles
            latencies = model_latency["latency"].values
            latencies = latencies[latencies > 0]  # Remove error cases
            
            if len(latencies) > 0:
                metrics["latency"] = {
                    "p50": float(np.percentile(latencies, 50)),
                    "p90": float(np.percentile(latencies, 90)),
                    "p95": float(np.percentile(latencies, 95)),
                    "mean": float(np.mean(latencies))
                }
            
            summary[model] = metrics
        
        return summary
    
    def generate_domain_breakdown(self, scores_df: pd.DataFrame, latency_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate domain-specific performance breakdown."""
        domain_summary = {}
        
        domains = scores_df["domain"].unique()
        
        for domain in domains:
            domain_data = {}
            
            for model in ["gpt-5", "claude-4-sonnet"]:
                model_scores = scores_df[(scores_df["model"] == model) & (scores_df["domain"] == domain)]
                model_latency = latency_df[(latency_df["model"] == model) & (latency_df["domain"] == domain)]
                
                if len(model_scores) == 0:
                    continue
                
                metrics = {}
                
                # Key metrics for this domain
                key_metrics = ["task_success", "factual_precision", "reasoning_quality", "helpfulness"]
                
                for metric in key_metrics:
                    if metric in model_scores.columns:
                        values = model_scores[metric].values
                        metrics[metric] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values))
                        }
                
                # Latency
                latencies = model_latency["latency"].values
                latencies = latencies[latencies > 0]
                
                if len(latencies) > 0:
                    metrics["latency_p50"] = float(np.percentile(latencies, 50))
                
                domain_data[model] = metrics
            
            domain_summary[domain] = domain_data
        
        return domain_summary
    
    def perform_statistical_tests(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        test_results = {}
        
        # Paired comparisons for each metric
        metrics = ["task_success", "factual_precision", "reasoning_quality", 
                  "helpfulness", "conciseness", "hallucination_rate"]
        
        for metric in metrics:
            gpt5_scores = scores_df[scores_df["model"] == "gpt-5"][metric].values
            claude4_scores = scores_df[scores_df["model"] == "claude-4-sonnet"][metric].values
            
            if len(gpt5_scores) == len(claude4_scores) and len(gpt5_scores) > 0:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(gpt5_scores, claude4_scores)
                
                # Effect size (Cohen's d for paired samples)
                diff = gpt5_scores - claude4_scores
                cohen_d = np.mean(diff) / np.std(diff)
                
                test_results[metric] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohen_d": float(cohen_d),
                    "significant": p_value < 0.05,
                    "gpt5_mean": float(np.mean(gpt5_scores)),
                    "claude4_mean": float(np.mean(claude4_scores))
                }
        
        return test_results
    
    def generate_summary_csv(self, scores_df: pd.DataFrame, latency_df: pd.DataFrame):
        """Generate summary CSV files."""
        # Overall summary
        overall_summary = []
        
        for model in ["gpt-5", "claude-4-sonnet"]:
            model_scores = scores_df[scores_df["model"] == model]
            model_latency = latency_df[latency_df["model"] == model]
            
            row = {"model": model}
            
            # Add metric means
            for metric in ["task_success", "factual_precision", "reasoning_quality", 
                          "helpfulness", "conciseness", "hallucination_rate", 
                          "safety_refusal_correctness"]:
                row[metric] = model_scores[metric].mean()
            
            # Add latency percentiles
            latencies = model_latency["latency"].values
            latencies = latencies[latencies > 0]
            
            if len(latencies) > 0:
                row["latency_p50"] = np.percentile(latencies, 50)
                row["latency_p90"] = np.percentile(latencies, 90)
                row["latency_p95"] = np.percentile(latencies, 95)
            
            overall_summary.append(row)
        
        pd.DataFrame(overall_summary).to_csv(self.results_dir / "summary.csv", index=False)
        
        # Domain breakdown
        domain_breakdown = []
        
        for domain in scores_df["domain"].unique():
            for model in ["gpt-5", "claude-4-sonnet"]:
                domain_scores = scores_df[(scores_df["model"] == model) & (scores_df["domain"] == domain)]
                domain_latency = latency_df[(latency_df["model"] == model) & (latency_df["domain"] == domain)]
                
                if len(domain_scores) == 0:
                    continue
                
                row = {"domain": domain, "model": model}
                
                for metric in ["task_success", "factual_precision", "reasoning_quality", "helpfulness"]:
                    row[metric] = domain_scores[metric].mean()
                
                latencies = domain_latency["latency"].values
                latencies = latencies[latencies > 0]
                if len(latencies) > 0:
                    row["latency_p50"] = np.percentile(latencies, 50)
                
                domain_breakdown.append(row)
        
        pd.DataFrame(domain_breakdown).to_csv(self.results_dir / "domain_breakdown.csv", index=False)
    
    def run_aggregation(self):
        """Run complete results aggregation."""
        print("Loading evaluation data...")
        
        # Load data
        scores_df = self.load_evaluation_scores()
        latency_df = self.load_latency_data()
        
        # Add domain information
        scores_df = self.add_domain_info(scores_df)
        
        print(f"Loaded {len(scores_df)} score records and {len(latency_df)} latency records")
        
        # Generate summaries
        print("Generating overall summary...")
        overall_summary = self.generate_overall_summary(scores_df, latency_df)
        
        print("Generating domain breakdown...")
        domain_breakdown = self.generate_domain_breakdown(scores_df, latency_df)
        
        print("Performing statistical tests...")
        statistical_tests = self.perform_statistical_tests(scores_df)
        
        # Save results
        results = {
            "overall_summary": overall_summary,
            "domain_breakdown": domain_breakdown,
            "statistical_tests": statistical_tests,
            "metadata": {
                "total_prompts": len(scores_df) // 2,  # Divided by 2 models
                "domains": list(scores_df["domain"].unique()),
                "models": ["gpt-5", "claude-4-sonnet"]
            }
        }
        
        with open(self.results_dir / "aggregated_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate CSV files
        print("Generating CSV summaries...")
        self.generate_summary_csv(scores_df, latency_df)
        
        print("Results aggregation complete!")
        print(f"Results saved to {self.results_dir}")

if __name__ == "__main__":
    aggregator = ResultsAggregator()
    aggregator.run_aggregation()
