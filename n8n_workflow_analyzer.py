#!/usr/bin/env python3
"""
n8n Workflow Data Analysis
Analyzes scraped workflow data to extract insights
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class N8nWorkflowAnalyzer:
    def __init__(self, data_file):
        """Initialize analyzer with scraped data"""
        self.data_file = Path(data_file)
        self.output_dir = self.data_file.parent.parent / 'analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading data from: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.workflows = json.load(f)
        
        print(f"Loaded {len(self.workflows)} workflows")
        self.df = self.create_dataframe()
    
    def create_dataframe(self):
        """Convert workflow data to pandas DataFrame"""
        records = []
        
        for workflow in self.workflows:
            # Extract node information
            node_names = [node.get('name', 'Unknown') for node in workflow.get('nodes', [])]
            node_types = [node.get('displayName', 'Unknown') for node in workflow.get('nodes', [])]
            node_categories = []
            for node in workflow.get('nodes', []):
                cats = [cat.get('name', '') for cat in node.get('nodeCategories', [])]
                node_categories.extend(cats)
            
            record = {
                'id': workflow.get('id'),
                'name': workflow.get('name'),
                'totalViews': workflow.get('totalViews', 0),
                'price': workflow.get('price', 0),
                'user_id': workflow.get('user', {}).get('id'),
                'user_name': workflow.get('user', {}).get('name'),
                'verified': workflow.get('user', {}).get('verified', False),
                'createdAt': workflow.get('createdAt'),
                'description': workflow.get('description', ''),
                'node_count': len(workflow.get('nodes', [])),
                'nodes': node_names,
                'node_types': node_types,
                'node_categories': node_categories,
                'purchaseUrl': workflow.get('purchaseUrl', '')
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert date
        if 'createdAt' in df.columns:
            df['createdAt'] = pd.to_datetime(df['createdAt'])
            df['created_year'] = df['createdAt'].dt.year
            df['created_month'] = df['createdAt'].dt.month
        
        return df
    
    def analyze_top_viewed(self, top_n=20):
        """Analyze most viewed workflows"""
        print("\n" + "=" * 60)
        print(f"TOP {top_n} MOST VIEWED WORKFLOWS")
        print("=" * 60)
        
        top_workflows = self.df.nlargest(top_n, 'totalViews')[
            ['name', 'totalViews', 'price', 'node_count', 'user_name']
        ]
        
        print(top_workflows.to_string(index=False))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        top_20 = self.df.nlargest(20, 'totalViews')
        
        bars = ax.barh(range(len(top_20)), top_20['totalViews'])
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels([name[:50] + '...' if len(name) > 50 else name 
                            for name in top_20['name']], fontsize=9)
        ax.set_xlabel('Total Views', fontsize=12)
        ax.set_title(f'Top {top_n} Most Viewed n8n Workflows', fontsize=14, fontweight='bold')
        
        # Color bars by price
        colors = plt.cm.viridis(top_20['price'] / top_20['price'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_viewed_workflows.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: top_viewed_workflows.png")
        
        return top_workflows
    
    def analyze_node_usage(self, top_n=30):
        """Analyze most frequently used nodes"""
        print("\n" + "=" * 60)
        print(f"TOP {top_n} MOST USED NODES")
        print("=" * 60)
        
        # Count node types
        all_node_types = []
        for nodes in self.df['node_types']:
            all_node_types.extend(nodes)
        
        node_counts = Counter(all_node_types)
        top_nodes = pd.DataFrame(node_counts.most_common(top_n), 
                                  columns=['Node', 'Usage Count'])
        
        print(top_nodes.to_string(index=False))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.barh(range(len(top_nodes)), top_nodes['Usage Count'], color='steelblue')
        ax.set_yticks(range(len(top_nodes)))
        ax.set_yticklabels(top_nodes['Node'], fontsize=10)
        ax.set_xlabel('Number of Workflows Using This Node', fontsize=12)
        ax.set_title(f'Top {top_n} Most Frequently Used n8n Nodes', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_nodes['Usage Count']):
            ax.text(v + 20, i, str(v), va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_used_nodes.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: top_used_nodes.png")
        
        return top_nodes
    
    def analyze_categories(self, top_n=15):
        """Analyze node categories (use cases)"""
        print("\n" + "=" * 60)
        print(f"TOP {top_n} NODE CATEGORIES (Use Cases)")
        print("=" * 60)
        
        # Count categories
        all_categories = []
        for cats in self.df['node_categories']:
            all_categories.extend(cats)
        
        category_counts = Counter(all_categories)
        top_categories = pd.DataFrame(category_counts.most_common(top_n), 
                                       columns=['Category', 'Usage Count'])
        
        print(top_categories.to_string(index=False))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(len(top_categories)), top_categories['Usage Count'], color='coral')
        ax.set_xticks(range(len(top_categories)))
        ax.set_xticklabels(top_categories['Category'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Number of Workflows', fontsize=12)
        ax.set_title(f'Top {top_n} Workflow Categories (Real Use Cases)', 
                     fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(top_categories['Usage Count']):
            ax.text(i, v + 30, str(v), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_categories.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: top_categories.png")
        
        return top_categories
    
    def analyze_pricing(self):
        """Analyze workflow pricing"""
        print("\n" + "=" * 60)
        print("PRICING ANALYSIS")
        print("=" * 60)
        
        free_workflows = len(self.df[self.df['price'] == 0])
        paid_workflows = len(self.df[self.df['price'] > 0])
        
        print(f"Free workflows: {free_workflows} ({free_workflows/len(self.df)*100:.1f}%)")
        print(f"Paid workflows: {paid_workflows} ({paid_workflows/len(self.df)*100:.1f}%)")
        
        if paid_workflows > 0:
            paid_df = self.df[self.df['price'] > 0]
            print(f"\nPaid Workflow Statistics:")
            print(f"  Min price: ${paid_df['price'].min():.2f}")
            print(f"  Max price: ${paid_df['price'].max():.2f}")
            print(f"  Average price: ${paid_df['price'].mean():.2f}")
            print(f"  Median price: ${paid_df['price'].median():.2f}")
            
            # Correlation between price and views
            if paid_df['totalViews'].sum() > 0:
                correlation = paid_df[['price', 'totalViews']].corr().iloc[0, 1]
                print(f"\nCorrelation between price and views: {correlation:.3f}")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        ax1.pie([free_workflows, paid_workflows], labels=['Free', 'Paid'],
                autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        ax1.set_title('Free vs Paid Workflows', fontsize=12, fontweight='bold')
        
        # Price distribution for paid workflows
        if paid_workflows > 0:
            ax2.hist(paid_df['price'], bins=30, color='steelblue', edgecolor='black')
            ax2.set_xlabel('Price ($)', fontsize=12)
            ax2.set_ylabel('Number of Workflows', fontsize=12)
            ax2.set_title('Price Distribution (Paid Workflows)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pricing_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: pricing_analysis.png")
    
    def analyze_complexity(self):
        """Analyze workflow complexity (node count)"""
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLEXITY ANALYSIS")
        print("=" * 60)
        
        print(f"Average nodes per workflow: {self.df['node_count'].mean():.1f}")
        print(f"Median nodes per workflow: {self.df['node_count'].median():.0f}")
        print(f"Min nodes: {self.df['node_count'].min()}")
        print(f"Max nodes: {self.df['node_count'].max()}")
        
        # Complexity categories
        self.df['complexity'] = pd.cut(self.df['node_count'], 
                                        bins=[0, 5, 10, 20, float('inf')],
                                        labels=['Simple (1-5)', 'Medium (6-10)', 
                                               'Complex (11-20)', 'Very Complex (20+)'])
        
        complexity_dist = self.df['complexity'].value_counts()
        print(f"\nComplexity Distribution:")
        for category, count in complexity_dist.items():
            print(f"  {category}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Correlation with views
        correlation = self.df[['node_count', 'totalViews']].corr().iloc[0, 1]
        print(f"\nCorrelation between complexity and views: {correlation:.3f}")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(self.df['node_count'], bins=30, color='green', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of Nodes', fontsize=12)
        ax1.set_ylabel('Number of Workflows', fontsize=12)
        ax1.set_title('Workflow Complexity Distribution', fontsize=12, fontweight='bold')
        ax1.axvline(self.df['node_count'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.df["node_count"].mean():.1f}')
        ax1.legend()
        
        # Scatter plot: complexity vs views
        ax2.scatter(self.df['node_count'], self.df['totalViews'], alpha=0.5, s=30)
        ax2.set_xlabel('Number of Nodes', fontsize=12)
        ax2.set_ylabel('Total Views', fontsize=12)
        ax2.set_title('Complexity vs Popularity', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: complexity_analysis.png")
    
    def analyze_creators(self, top_n=15):
        """Analyze top workflow creators"""
        print("\n" + "=" * 60)
        print(f"TOP {top_n} WORKFLOW CREATORS")
        print("=" * 60)
        
        creator_stats = self.df.groupby('user_name').agg({
            'id': 'count',
            'totalViews': 'sum',
            'verified': 'first'
        }).rename(columns={'id': 'workflow_count', 'totalViews': 'total_views'})
        
        creator_stats = creator_stats.sort_values('workflow_count', ascending=False).head(top_n)
        print(creator_stats.to_string())
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        creators = creator_stats.index[:top_n]
        counts = creator_stats['workflow_count'][:top_n]
        
        bars = ax.barh(range(len(creators)), counts)
        ax.set_yticks(range(len(creators)))
        ax.set_yticklabels(creators, fontsize=10)
        ax.set_xlabel('Number of Workflows Created', fontsize=12)
        ax.set_title(f'Top {top_n} Most Prolific Workflow Creators', 
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Color verified users differently
        colors = ['gold' if creator_stats.loc[creator, 'verified'] else 'steelblue' 
                  for creator in creators]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, v in enumerate(counts):
            ax.text(v + 0.5, i, str(v), va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_creators.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: top_creators.png")
        
        return creator_stats
    
    def generate_full_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("GENERATING FULL ANALYSIS REPORT")
        print("=" * 70)
        
        report = {
            'summary_statistics': {
                'total_workflows': int(len(self.df)),
                'total_views': int(self.df['totalViews'].sum()),
                'avg_views_per_workflow': float(self.df['totalViews'].mean()),
                'unique_creators': int(self.df['user_name'].nunique()),
                'avg_nodes_per_workflow': float(self.df['node_count'].mean()),
                'free_workflows': int(len(self.df[self.df['price'] == 0])),
                'paid_workflows': int(len(self.df[self.df['price'] > 0])),
            }
        }
        
        # Run all analyses and convert to JSON-serializable format
        report['top_viewed'] = self.analyze_top_viewed(20).to_dict('records')
        report['top_nodes'] = self.analyze_node_usage(30).to_dict('records')
        report['top_categories'] = self.analyze_categories(15).to_dict('records')
        self.analyze_pricing()
        self.analyze_complexity()
        
        # Convert top_creators DataFrame to JSON-serializable format
        top_creators_df = self.analyze_creators(15)
        report['top_creators'] = []
        for idx, row in top_creators_df.iterrows():
            report['top_creators'].append({
                'creator_name': str(idx),
                'workflow_count': int(row['workflow_count']),
                'total_views': int(row['total_views']),
                'verified': bool(row['verified'])
            })
        
        # Save report
        report_file = self.output_dir / 'analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'=' * 70}")
        print(f"✓ Full analysis complete!")
        print(f"✓ Results saved to: {self.output_dir}")
        print(f"✓ Report saved to: {report_file}")
        print(f"{'=' * 70}\n")
        
        return report


def main():
    """Main execution"""
    # Find the most recent workflow data file
    data_dir = Path('n8n_data/raw')
    
    if not data_dir.exists():
        print("Error: No data directory found. Run the scraper first!")
        return
    
    workflow_files = list(data_dir.glob('workflows_*.json'))
    
    if not workflow_files:
        print("Error: No workflow data files found. Run the scraper first!")
        return
    
    # Use the most recent file
    latest_file = max(workflow_files, key=lambda p: p.stat().st_mtime)
    print(f"Using data file: {latest_file}")
    
    # Run analysis
    analyzer = N8nWorkflowAnalyzer(latest_file)
    analyzer.generate_full_report()


if __name__ == '__main__':
    main()