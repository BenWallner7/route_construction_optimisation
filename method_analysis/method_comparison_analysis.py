import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
import json
from typing import Dict, List
import os

# Load results csv produced by `evaluate_methods.py`
results_df = pd.read_csv('route_opt_results.csv')

output_dir = '/home/benwallner7/route_construction_optimisation/method_analysis'

# Load instances for additional analysis

with open('/home/benwallner7/route_construction_optimisation/instances.json', 'r') as f:
    instances = json.load(f)



total_goals_by_method = results_df.groupby('method')['goals_visited'].sum().sort_values(ascending=False)
runtime_summary = results_df.groupby('method')
methods = results_df['method'].unique()

# Add instance characteristics
instance_chars = []
for instance_name in results_df['instance'].unique():
    if instance_name in instances:
        inst_data = instances[instance_name]
        instance_chars.append({
            'instance': instance_name,
            'num_starts': len(inst_data['start_points']),
            'num_ends': len(inst_data['end_points']),
            'num_goals': len(inst_data['goal_points']),
            'total_points': len(inst_data['start_points']) + len(inst_data['end_points']) + len(inst_data['goal_points'])
        })

instance_chars_df = pd.DataFrame(instance_chars)

# Merge with results
results_with_chars = results_df.merge(instance_chars_df, on='instance', how='left')


# Rank methods by different criteria
ranking_criteria = {
    'Total Goals': results_df.groupby('method')['goals_visited'].sum(),
    'Avg Goals': results_df.groupby('method')['goals_visited'].mean(),
    'Speed (1/runtime)': 1 / results_df.groupby('method')['runtime'].mean(),
    'Reliability': results_df.groupby('method')['valid'].mean(),
    'Efficiency': (results_df['goals_visited'] / results_df['num_points']).groupby(results_df['method']).mean()
}


# Overall performance score - equal weighting
weights = {'goals': 0.4, 'speed': 0.2, 'reliability': 0.2, 'efficiency': 0.2}
overall_scores = {}

for method in methods:
    method_data = results_df[results_df['method'] == method]
    
    # Normalize each metric to 0-1 scale
    goals_norm = method_data['goals_visited'].mean() / results_df['goals_visited'].max()
    speed_norm = results_df['runtime'].min() / method_data['runtime'].mean()  # Inverse
    reliability_norm = method_data['valid'].mean()
    efficiency_norm = (method_data['goals_visited'] / method_data['num_points']).mean()
    efficiency_norm = efficiency_norm / (results_df['goals_visited'] / results_df['num_points']).max()
    
    overall_scores[method] = (
        weights['goals'] * goals_norm +
        weights['speed'] * speed_norm +
        weights['reliability'] * reliability_norm +
        weights['efficiency'] * efficiency_norm
    )
    
sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)



# Generate analysis report
report_content = f"""Route Optimization Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}


=== EXECUTIVE SUMMARY ===
Total Experiments: {len(results_df)}
Methods Compared: {', '.join(methods)}
Best Overall Method: {sorted_overall[0][0]} (score: {sorted_overall[0][1]:.4f})

=== PRIMARY METRICS ===
Total Goals Visited by Method:
"""
    
for method, total in total_goals_by_method.items():
    count = results_df[results_df['method'] == method].shape[0]
    avg = total / count if count > 0 else 0
    report_content += f"  {method}: {total} total ({avg:.2f} avg)\n"
    
report_content += "\nConstraint Validation:\n"

# Compute group summary
valid_summary = results_df.groupby('method')['valid'].agg([
    'sum',
    'count',
    lambda x: (x.sum() / len(x) * 100)
])
valid_summary.columns = ['valid_solutions', 'total_attempts', 'success_rate_pct']

# Append each method’s validation stats to the report content
for method, row in valid_summary.iterrows():
    report_content += (
        f"{method:>10s}: "
        f"{int(row['valid_solutions'])}/"
        f"{int(row['total_attempts'])} valid "
        f"({row['success_rate_pct']:.1f}%)\n"
    )

report_content += f"""
Runtime Performance:
"""
runtime_summary = results_df.groupby('method')['runtime'].agg(['sum', 'mean', 'std']).round(6).reset_index()
for _, row in runtime_summary.iterrows():
    method = row['method']
    report_content += f"  {method}: {row['mean']:.6f}s avg, {row['sum']:.4f}s total\n"

report_content += f"""
Instance Level Analysis:
"""
# Loop through each characteristic and compute correlations
for char in ['num_starts', 'num_ends', 'num_goals']:
    if char in results_with_chars.columns:
        corr = results_with_chars['goals_visited'].corr(results_with_chars[char])
        report_content += f"  Goals visited vs {char}: correlation = {corr:.3f}\n"
    
report_content += f"""
=== METHOD RANKINGS ===
"""
for criterion, values in ranking_criteria.items():
    ranked = values.sort_values(ascending=False)
    report_content += f"\n{criterion}:\n"
    for i, (method, score) in enumerate(ranked.items(), 1):
        report_content += f"  {i}. {method}: {score:.4f}\n"

report_content += f"""
=== RECOMMENDATIONS ===
- For maximum goal coverage: {ranking_criteria['Total Goals'].idxmax()}
- For fastest execution: {ranking_criteria['Speed (1/runtime)'].idxmax()}
- For best reliability: {ranking_criteria['Reliability'].idxmax()}
- For best efficiency: {ranking_criteria['Efficiency'].idxmax()}
- Best overall balance: {sorted_overall[0][0]}
"""

# Save report
with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
    f.write(report_content)