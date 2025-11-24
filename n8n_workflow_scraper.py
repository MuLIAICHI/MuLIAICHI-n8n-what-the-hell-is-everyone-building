#!/usr/bin/env python3
"""
n8n Workflow Scraper
Collects workflow data from n8n marketplace for analysis
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class N8nWorkflowScraper:
    def __init__(self, output_dir='n8n_data'):
        self.base_url = "https://n8n.io/api/product-api/workflows/search"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            
        })
    
    def fetch_page(self, category=None, rows=50, page=1, delay=1.5):
        """
        Fetch a single page of workflows
        
        Args:
            category: Filter by category (AI, Marketing, etc.) or None for all
            rows: Number of results per page (max seems to be 50-100)
            page: Page number
            delay: Delay between requests in seconds
        """
        params = {
            'rows': rows,
            'page': page
        }
        
        if category:
            params['category'] = category
        
        try:
            logger.info(f"Fetching page {page} (category: {category or 'all'})")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            time.sleep(delay)  # Rate limiting
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {page}: {e}")
            return None
    
    def scrape_all_workflows(self, category=None, rows_per_page=50, max_pages=None):
        """
        Scrape all workflows with pagination
        
        Args:
            category: Filter by category or None for all
            rows_per_page: Results per page
            max_pages: Maximum pages to scrape (None for all)
        """
        all_workflows = []
        page = 1
        total_workflows = None
        
        while True:
            # Check if we've hit max_pages
            if max_pages and page > max_pages:
                logger.info(f"Reached max_pages limit: {max_pages}")
                break
            
            data = self.fetch_page(category, rows_per_page, page)
            
            if not data:
                logger.warning(f"No data returned for page {page}")
                break
            
            # Get total count on first page
            if page == 1:
                total_workflows = data.get('totalWorkflows', 0)
                logger.info(f"Total workflows available: {total_workflows}")
            
            workflows = data.get('workflows', [])
            
            if not workflows:
                logger.info(f"No more workflows found at page {page}")
                break
            
            all_workflows.extend(workflows)
            logger.info(f"Page {page}: Collected {len(workflows)} workflows (Total: {len(all_workflows)})")
            
            # Check if we've collected all available workflows
            if len(all_workflows) >= total_workflows:
                logger.info("Collected all available workflows")
                break
            
            page += 1
        
        return all_workflows, data.get('filters', [])
    
    def scrape_by_categories(self, categories=None, rows_per_page=50):
        """
        Scrape workflows by category to ensure complete coverage
        
        Args:
            categories: List of categories or None to fetch from first request
        """
        # If no categories provided, get them from the first request
        if not categories:
            logger.info("Fetching available categories...")
            first_page = self.fetch_page(rows=1, page=1)
            if first_page and 'filters' in first_page:
                category_filter = next((f for f in first_page['filters'] if f['field_name'] == 'categories'), None)
                if category_filter:
                    categories = [c['value'] for c in category_filter['counts'][:10]]  # Top 10 categories
                    logger.info(f"Found categories: {categories}")
        
        all_workflows = []
        all_filters = {}
        
        # Scrape all workflows without category filter first
        logger.info("=" * 50)
        logger.info("Scraping ALL workflows (no category filter)")
        logger.info("=" * 50)
        workflows, filters = self.scrape_all_workflows(category=None, rows_per_page=rows_per_page)
        all_workflows.extend(workflows)
        all_filters['all'] = filters
        
        # Also scrape by category to ensure we don't miss anything
        if categories:
            for category in categories:
                logger.info("=" * 50)
                logger.info(f"Scraping category: {category}")
                logger.info("=" * 50)
                workflows, filters = self.scrape_all_workflows(category=category, rows_per_page=rows_per_page)
                all_filters[category] = filters
                
                # Add workflows, avoiding duplicates
                existing_ids = {w['id'] for w in all_workflows}
                new_workflows = [w for w in workflows if w['id'] not in existing_ids]
                all_workflows.extend(new_workflows)
                logger.info(f"Added {len(new_workflows)} new workflows from {category}")
        
        return all_workflows, all_filters
    
    def save_data(self, workflows, filters, prefix=''):
        """Save scraped data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        workflows_file = self.output_dir / 'raw' / f'{prefix}workflows_{timestamp}.json'
        with open(workflows_file, 'w', encoding='utf-8') as f:
            json.dump(workflows, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(workflows)} workflows to {workflows_file}")
        
        # Save filters
        filters_file = self.output_dir / 'raw' / f'{prefix}filters_{timestamp}.json'
        with open(filters_file, 'w', encoding='utf-8') as f:
            json.dump(filters, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved filters to {filters_file}")
        
        # Save summary statistics
        summary = {
            'timestamp': timestamp,
            'total_workflows': len(workflows),
            'unique_users': len(set(w['user']['id'] for w in workflows if 'user' in w)),
            'categories': list(set(cat for w in workflows for cat in w.get('categories', []))),
            'total_views': sum(w.get('totalViews', 0) for w in workflows),
            'price_range': {
                'min': min((w.get('price', 0) for w in workflows), default=0),
                'max': max((w.get('price', 0) for w in workflows), default=0),
                'avg': sum(w.get('price', 0) for w in workflows) / len(workflows) if workflows else 0
            }
        }
        
        summary_file = self.output_dir / f'{prefix}summary_{timestamp}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        return workflows_file, filters_file, summary_file


def main():
    """Main execution"""
    logger.info("Starting n8n Workflow Scraper")
    logger.info("=" * 50)
    
    scraper = N8nWorkflowScraper()
    
    # Option 1: Scrape all workflows
    logger.info("Scraping all workflows...")
    workflows, filters = scraper.scrape_all_workflows(rows_per_page=50)
    
    # Option 2: Scrape by categories for complete coverage
    # workflows, filters = scraper.scrape_by_categories(rows_per_page=50)
    
    # Save data
    logger.info("=" * 50)
    logger.info("Saving data...")
    scraper.save_data(workflows, filters)
    
    logger.info("=" * 50)
    logger.info(f"✓ Scraping complete! Total workflows: {len(workflows)}")
    logger.info(f"Data saved to: {scraper.output_dir}")


if __name__ == '__main__':
    main()
