#!/usr/bin/env python3
"""
All-in-One: Scrape & Analyze n8n Workflows
One command to rule them all!
"""

import sys
import time
from pathlib import Path

def main():
    print("=" * 70)
    print("🚀 n8n WORKFLOW SCRAPER & ANALYZER")
    print("=" * 70)
    print("\nThis will:")
    print("1. Scrape all workflows from n8n marketplace")
    print("2. Analyze the data and generate insights")
    print("3. Create visualizations")
    print("\nEstimated time: 5-10 minutes")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Scrape
    print("\n" + "=" * 70)
    print("STEP 1: SCRAPING WORKFLOWS")
    print("=" * 70)
    
    try:
        from n8n_workflow_scraper import N8nWorkflowScraper
        
        scraper = N8nWorkflowScraper()
        workflows, filters = scraper.scrape_all_workflows(rows_per_page=50)
        
        if not workflows:
            print("❌ No workflows found. Exiting.")
            return
        
        workflow_file, filters_file, summary_file = scraper.save_data(workflows, filters)
        
        print(f"\n✓ Scraping complete!")
        print(f"✓ Collected {len(workflows)} workflows")
        
        # Brief pause
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return
    
    # Step 2: Analyze
    print("\n" + "=" * 70)
    print("STEP 2: ANALYZING DATA")
    print("=" * 70)
    
    try:
        from n8n_workflow_analyzer import N8nWorkflowAnalyzer
        
        analyzer = N8nWorkflowAnalyzer(workflow_file)
        report = analyzer.generate_full_report()
        
        print("\n" + "=" * 70)
        print("🎉 ALL DONE!")
        print("=" * 70)
        print(f"\nResults saved to:")
        print(f"  - Raw data: n8n_data/raw/")
        print(f"  - Analysis: n8n_data/analysis/")
        print(f"  - Visualizations: n8n_data/analysis/*.png")
        
        print("\n📊 Quick Summary:")
        print(f"  Total workflows analyzed: {report['summary_statistics']['total_workflows']}")
        print(f"  Total views: {report['summary_statistics']['total_views']:,}")
        print(f"  Unique creators: {report['summary_statistics']['unique_creators']}")
        print(f"  Average nodes per workflow: {report['summary_statistics']['avg_nodes_per_workflow']:.1f}")
        
        print("\n✨ Check out the visualizations in n8n_data/analysis/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
