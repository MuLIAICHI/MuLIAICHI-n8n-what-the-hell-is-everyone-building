#!/usr/bin/env python3
"""
Advanced n8n Workflow Analysis
Deeper insights for power users
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np

class AdvancedN8nAnalyzer:
    def __init__(self, data_file):
        """Initialize advanced analyzer"""
        self.data_file = Path(data_file)
        self.output_dir = self.data_file.parent.parent / 'analysis_advanced'
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading data from: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.workflows = json.load(f)
        
        self.df = self.create_dataframe()
        print(f"Loaded {len(self.workflows)} workflows")
    
    def create_dataframe(self):
        """Convert to DataFrame with advanced fields"""
        records = []
        
        for workflow in self.workflows:
            node_types = [node.get('displayName', 'Unknown') for node in workflow.get('nodes', [])]
            
            record = {
                'id': workflow.get('id'),
                'name': workflow.get('name'),
                'totalViews': workflow.get('totalViews', 0),
                'price': workflow.get('price', 0),
                'node_count': len(workflow.get('nodes', [])),
                'node_types': node_types,
                'description': workflow.get('description', ''),
                'user_id': workflow.get('user', {}).get('id'),
                'user_name': workflow.get('user', {}).get('name'),
                'verified': workflow.get('user', {}).get('verified', False),
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def analyze_node_combinations(self, min_support=20, top_n=20):
        """Find common node combinations (which nodes are used together)"""
        print("\n" + "=" * 70)
        print(f"TOP {top_n} NODE COMBINATIONS (Co-occurrence Analysis)")
        print("=" * 70)
        
        # Count 2-node combinations
        combo_counts = Counter()
        
        for nodes in self.df['node_types']:
            if len(nodes) >= 2:
                # Get all unique pairs
                for combo in combinations(sorted(set(nodes)), 2):
                    combo_counts[combo] += 1
        
        # Filter by minimum support and get top combinations
        popular_combos = {k: v for k, v in combo_counts.items() if v >= min_support}
        top_combos = sorted(popular_combos.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\nFound {len(popular_combos)} combinations with {min_support}+ occurrences")
        print(f"\nTop {top_n} node pairs:")
        for (node1, node2), count in top_combos:
            print(f"  {node1} + {node2}: {count} workflows")
        
        # Visualize as heatmap
        self._visualize_node_network(top_combos)
        
        return top_combos
    
    def _visualize_node_network(self, combos):
        """Visualize node combinations"""
        # Get unique nodes from top combos
        nodes = set()
        for (n1, n2), count in combos[:30]:
            nodes.add(n1)
            nodes.add(n2)
        
        nodes = sorted(list(nodes))
        n = len(nodes)
        
        # Create adjacency matrix
        matrix = np.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for (n1, n2), count in combos:
            if n1 in node_to_idx and n2 in node_to_idx:
                i, j = node_to_idx[n1], node_to_idx[n2]
                matrix[i, j] = count
                matrix[j, i] = count
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(matrix, annot=False, cmap='YlOrRd', 
                    xticklabels=nodes, yticklabels=nodes, ax=ax)
        ax.set_title('Node Co-occurrence Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'node_combinations_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: node_combinations_heatmap.png")
    
    def analyze_success_factors(self):
        """Analyze what makes a workflow successful (high views)"""
        print("\n" + "=" * 70)
        print("SUCCESS FACTORS ANALYSIS")
        print("=" * 70)
        
        # Define success threshold (top 25%)
        success_threshold = self.df['totalViews'].quantile(0.75)
        self.df['successful'] = self.df['totalViews'] >= success_threshold
        
        successful = self.df[self.df['successful']]
        unsuccessful = self.df[~self.df['successful']]
        
        print(f"\nSuccess threshold: {success_threshold:.0f} views")
        print(f"Successful workflows: {len(successful)} ({len(successful)/len(self.df)*100:.1f}%)")
        
        # Compare characteristics
        factors = {
            'Average node count': {
                'Successful': successful['node_count'].mean(),
                'Others': unsuccessful['node_count'].mean()
            },
            'Average price': {
                'Successful': successful['price'].mean(),
                'Others': unsuccessful['price'].mean()
            },
            'Verified creators (%)': {
                'Successful': successful['verified'].sum() / len(successful) * 100,
                'Others': unsuccessful['verified'].sum() / len(unsuccessful) * 100
            }
        }
        
        print("\nComparative Analysis:")
        for factor, values in factors.items():
            print(f"\n{factor}:")
            for group, value in values.items():
                print(f"  {group}: {value:.2f}")
        
        # Most common nodes in successful workflows
        successful_nodes = []
        for nodes in successful['node_types']:
            successful_nodes.extend(nodes)
        
        node_counts = Counter(successful_nodes).most_common(15)
        
        print(f"\nMost common nodes in successful workflows:")
        for node, count in node_counts:
            percentage = count / len(successful) * 100
            print(f"  {node}: {count} ({percentage:.1f}%)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Factor comparison
        factor_names = ['Avg Node\nCount', 'Avg Price\n($)', 'Verified\nCreators (%)']
        successful_vals = [factors[k]['Successful'] for k in factors.keys()]
        other_vals = [factors[k]['Others'] for k in factors.keys()]
        
        x = np.arange(len(factor_names))
        width = 0.35
        
        ax1.bar(x - width/2, successful_vals, width, label='Successful', color='green', alpha=0.7)
        ax1.bar(x + width/2, other_vals, width, label='Others', color='gray', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(factor_names)
        ax1.set_title('Success Factor Comparison', fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Top nodes in successful workflows
        nodes = [n for n, c in node_counts[:10]]
        counts = [c for n, c in node_counts[:10]]
        
        ax2.barh(range(len(nodes)), counts, color='green', alpha=0.7)
        ax2.set_yticks(range(len(nodes)))
        ax2.set_yticklabels(nodes, fontsize=9)
        ax2.set_xlabel('Usage Count in Successful Workflows')
        ax2.set_title('Key Nodes for Success', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_factors.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: success_factors.png")
    
    def analyze_workflow_patterns(self):
        """Identify common workflow patterns"""
        print("\n" + "=" * 70)
        print("WORKFLOW PATTERN ANALYSIS")
        print("=" * 70)
        
        # Common workflow starting nodes
        start_nodes = []
        for nodes in self.df['node_types']:
            if nodes:
                start_nodes.append(nodes[0])
        
        start_counts = Counter(start_nodes).most_common(10)
        
        print("\nMost common workflow starting points:")
        for node, count in start_counts:
            print(f"  {node}: {count} workflows ({count/len(self.df)*100:.1f}%)")
        
        # Common ending nodes
        end_nodes = []
        for nodes in self.df['node_types']:
            if nodes:
                end_nodes.append(nodes[-1])
        
        end_counts = Counter(end_nodes).most_common(10)
        
        print("\nMost common workflow endpoints:")
        for node, count in end_counts:
            print(f"  {node}: {count} workflows ({count/len(self.df)*100:.1f}%)")
        
        # Workflow archetypes based on length
        self.df['archetype'] = pd.cut(
            self.df['node_count'],
            bins=[0, 3, 7, 15, float('inf')],
            labels=['Micro (1-3)', 'Small (4-7)', 'Medium (8-15)', 'Large (15+)']
        )
        
        archetype_stats = self.df.groupby('archetype').agg({
            'id': 'count',
            'totalViews': ['mean', 'median', 'sum']
        }).round(1)
        
        print("\nWorkflow Archetypes:")
        print(archetype_stats)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Starting nodes
        nodes = [n for n, c in start_counts]
        counts = [c for n, c in start_counts]
        ax1.barh(range(len(nodes)), counts, color='skyblue')
        ax1.set_yticks(range(len(nodes)))
        ax1.set_yticklabels(nodes, fontsize=9)
        ax1.set_xlabel('Number of Workflows')
        ax1.set_title('Most Common Starting Nodes', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Ending nodes
        nodes = [n for n, c in end_counts]
        counts = [c for n, c in end_counts]
        ax2.barh(range(len(nodes)), counts, color='lightcoral')
        ax2.set_yticks(range(len(nodes)))
        ax2.set_yticklabels(nodes, fontsize=9)
        ax2.set_xlabel('Number of Workflows')
        ax2.set_title('Most Common Ending Nodes', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'workflow_patterns.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: workflow_patterns.png")
    
    def analyze_description_keywords(self, top_n=30):
        """Analyze common keywords in workflow descriptions"""
        print("\n" + "=" * 70)
        print(f"TOP {top_n} DESCRIPTION KEYWORDS")
        print("=" * 70)
        
        from collections import Counter
        import re
        
        # Extract all words from descriptions
        all_words = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'is',
                      'it', 'this', 'that', 'you', 'your', 'can', 'be', 'are', 'will', 'as'}
        
        for desc in self.df['description']:
            if desc:
                # Extract words (3+ characters, alphabetic)
                words = re.findall(r'\b[a-z]{3,}\b', desc.lower())
                words = [w for w in words if w not in stop_words]
                all_words.extend(words)
        
        keyword_counts = Counter(all_words).most_common(top_n)
        
        print("\nMost frequent keywords in descriptions:")
        for keyword, count in keyword_counts:
            print(f"  {keyword}: {count}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        keywords = [k for k, c in keyword_counts]
        counts = [c for k, c in keyword_counts]
        
        ax.barh(range(len(keywords)), counts, color='purple', alpha=0.6)
        ax.set_yticks(range(len(keywords)))
        ax.set_yticklabels(keywords, fontsize=10)
        ax.set_xlabel('Frequency in Descriptions', fontsize=12)
        ax.set_title(f'Top {top_n} Keywords in Workflow Descriptions', 
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'description_keywords.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: description_keywords.png")
        
        return keyword_counts
    
    def generate_advanced_report(self):
        """Generate full advanced analysis report"""
        print("\n" + "=" * 70)
        print("GENERATING ADVANCED ANALYSIS REPORT")
        print("=" * 70)
        
        # Run all advanced analyses
        combos = self.analyze_node_combinations(min_support=20, top_n=20)
        self.analyze_success_factors()
        self.analyze_workflow_patterns()
        keywords = self.analyze_description_keywords(top_n=30)
        
        # Save report
        report = {
            'node_combinations': [{'nodes': list(combo), 'count': count} 
                                  for combo, count in combos],
            'top_keywords': [{'keyword': k, 'count': c} for k, c in keywords]
        }
        
        report_file = self.output_dir / 'advanced_analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'=' * 70}")
        print(f"✓ Advanced analysis complete!")
        print(f"✓ Results saved to: {self.output_dir}")
        print(f"{'=' * 70}\n")


def main():
    """Main execution"""
    data_dir = Path('n8n_data/raw')
    
    if not data_dir.exists():
        print("Error: No data directory found. Run the scraper first!")
        return
    
    workflow_files = list(data_dir.glob('workflows_*.json'))
    
    if not workflow_files:
        print("Error: No workflow data files found. Run the scraper first!")
        return
    
    latest_file = max(workflow_files, key=lambda p: p.stat().st_mtime)
    print(f"Using data file: {latest_file}")
    
    analyzer = AdvancedN8nAnalyzer(latest_file)
    analyzer.generate_advanced_report()


if __name__ == '__main__':
    main()
