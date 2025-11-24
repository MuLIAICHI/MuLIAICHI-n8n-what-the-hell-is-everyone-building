# n8n Workflow Scraper & Analyzer 🚀

A comprehensive tool to scrape and analyze 1000+ n8n workflows from the n8n marketplace to discover trends, popular nodes, and real-world use cases.

## 📋 Features

### Data Collection (Scraper)
- ✅ Scrapes all workflows from n8n marketplace
- ✅ Smart pagination handling
- ✅ Rate limiting to avoid API issues
- ✅ Category-based scraping for complete coverage
- ✅ Saves raw data in JSON format
- ✅ Automatic summary statistics

### Data Analysis (Analyzer)
- 📊 **Most Viewed Workflows** - Discover what's popular
- 🔧 **Most Used Nodes** - See which nodes are essential
- 🎯 **Use Case Categories** - Understand real-world applications
- 💰 **Pricing Analysis** - Free vs paid workflow insights
- 🧩 **Complexity Analysis** - Node count distribution and trends
- 👥 **Top Creators** - Most prolific workflow builders
- 📈 **Beautiful Visualizations** - Charts and graphs for all metrics

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Scraper

```bash
python n8n_workflow_scraper.py
```

This will:
- Scrape all available workflows (4600+)
- Save raw data to `n8n_data/raw/`
- Generate summary statistics
- Take approximately 5-10 minutes depending on API response time

### 3. Run the Analysis

```bash
python n8n_workflow_analyzer.py
```

This will:
- Process the scraped data
- Generate comprehensive analysis
- Create visualizations as PNG files
- Save results to `n8n_data/analysis/`

## 📂 Project Structure

```
n8n-workflow-analysis/
├── n8n_workflow_scraper.py      # Data collection script
├── n8n_workflow_analyzer.py     # Data analysis script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── n8n_data/                     # Output directory (created automatically)
    ├── raw/                      # Raw scraped data
    │   ├── workflows_YYYYMMDD_HHMMSS.json
    │   └── filters_YYYYMMDD_HHMMSS.json
    ├── processed/                # Processed data
    ├── analysis/                 # Analysis results
    │   ├── analysis_report.json
    │   ├── top_viewed_workflows.png
    │   ├── top_used_nodes.png
    │   ├── top_categories.png
    │   ├── pricing_analysis.png
    │   ├── complexity_analysis.png
    │   └── top_creators.png
    └── summary_YYYYMMDD_HHMMSS.json
```

## 🔧 Advanced Usage

### Scraper Options

#### Scrape Specific Category
```python
from n8n_workflow_scraper import N8nWorkflowScraper

scraper = N8nWorkflowScraper()
workflows, filters = scraper.scrape_all_workflows(
    category='AI',          # Filter by category
    rows_per_page=50,       # Results per page
    max_pages=20            # Limit number of pages
)
scraper.save_data(workflows, filters, prefix='ai_')
```

#### Scrape Multiple Categories
```python
categories = ['AI', 'Marketing', 'Sales', 'Support']
workflows, filters = scraper.scrape_by_categories(
    categories=categories,
    rows_per_page=50
)
scraper.save_data(workflows, filters, prefix='multi_')
```

### Analyzer Options

#### Custom Analysis
```python
from n8n_workflow_analyzer import N8nWorkflowAnalyzer

# Initialize with specific data file
analyzer = N8nWorkflowAnalyzer('n8n_data/raw/workflows_20250109_120000.json')

# Run individual analyses
top_workflows = analyzer.analyze_top_viewed(top_n=50)
top_nodes = analyzer.analyze_node_usage(top_n=100)
categories = analyzer.analyze_categories(top_n=20)

# Or generate full report
report = analyzer.generate_full_report()
```

## 📊 Sample Outputs

### Top Viewed Workflows
```
TOP 20 MOST VIEWED WORKFLOWS
============================================================
name                                               totalViews  price  node_count  user_name
AI Video Automation Engine - Generate & Pub...           1234     10          23  John Doe
Customer Support Chatbot with GPT-4                       987     15          18  Jane Smith
...
```

### Top Used Nodes
```
TOP 30 MOST USED NODES
============================================================
Node                           Usage Count
Sticky Note                           4315
Edit Fields (Set)                     2318
HTTP Request                          2277
Code                                  2166
AI Agent                              2120
...
```

### Node Categories (Real Use Cases)
```
TOP 15 NODE CATEGORIES (Use Cases)
============================================================
Category                       Usage Count
AI                                    4617
Multimodal AI                         2498
Marketing                             1717
AI Summarization                      1294
Content Creation                      1081
...
```

## 📈 Analysis Insights You'll Get

1. **Popularity Metrics**
   - Which workflows get the most views
   - Correlation between complexity and popularity
   - Price vs views analysis

2. **Node Usage Patterns**
   - Essential nodes every workflow uses
   - Specialized nodes for specific use cases
   - Node combinations that work well together

3. **Use Case Distribution**
   - Most common workflow categories
   - Emerging trends (AI, automation, etc.)
   - Niche applications

4. **Creator Insights**
   - Most prolific workflow creators
   - Verified vs community creators
   - Quality indicators

5. **Complexity Analysis**
   - Simple vs complex workflows
   - Average node count
   - Complexity sweet spots

6. **Pricing Trends**
   - Free vs paid distribution
   - Price ranges and averages
   - Value indicators

## 🎯 Use Cases for This Tool

### For n8n Content Creators (Like You!)
- **Content Ideas**: See what topics get the most views
- **Tutorial Planning**: Focus on most-used nodes
- **Market Gaps**: Find underserved categories
- **Pricing Strategy**: Understand market pricing

### For n8n Users
- **Learning Path**: Start with most common nodes
- **Best Practices**: Study popular workflows
- **Use Case Discovery**: Find workflows for your needs
- **Quality Signals**: Identify trusted creators

### For Business/Product Analysis
- **Market Research**: Understand n8n ecosystem
- **Feature Priorities**: See what users build most
- **Integration Opportunities**: Popular tool combinations
- **Competitive Analysis**: Compare your workflows

## 🔍 API Endpoint Details

The scraper uses the n8n marketplace API:
```
https://n8n.io/api/product-api/workflows/search
```

### Parameters:
- `category` - Filter by category (optional)
- `rows` - Results per page (1-100)
- `page` - Page number (1-N)

### Response Structure:
```json
{
  "totalWorkflows": 4617,
  "workflows": [...],
  "filters": [...]
}
```

## ⚠️ Important Notes

1. **Rate Limiting**: The scraper includes 1.5s delays between requests to be respectful to the API
2. **Data Size**: Full scrape generates ~50-100MB of JSON data
3. **Processing Time**: 
   - Scraping: 5-10 minutes for all workflows
   - Analysis: 30-60 seconds
4. **Updates**: Re-run scraper periodically to get latest data

## 🐛 Troubleshooting

### Connection Issues
```python
# Increase delay between requests
scraper.fetch_page(delay=3.0)  # 3 seconds instead of 1.5
```

### Memory Issues
```python
# Scrape in smaller batches
workflows, filters = scraper.scrape_all_workflows(max_pages=50)
```

### Analysis Errors
```bash
# Make sure you ran scraper first
ls -la n8n_data/raw/

# Check data file exists
python -c "import json; json.load(open('n8n_data/raw/workflows_*.json'))"
```

## 📝 Data Fields Reference

Each workflow contains:
- `id` - Unique identifier
- `name` - Workflow title
- `totalViews` - View count
- `price` - Price in USD (0 for free)
- `user` - Creator information
- `description` - Full description
- `nodes` - Array of node objects
- `createdAt` - Creation timestamp
- `purchaseUrl` - Gumroad link

## 🚀 Next Steps & Ideas

1. **Time-Series Analysis**: Track trends over time
2. **Node Relationships**: Which nodes are commonly used together
3. **Success Prediction**: ML model to predict workflow popularity
4. **Recommendation Engine**: Suggest workflows based on user interests
5. **Quality Scoring**: Automated quality assessment
6. **Community Insights**: Creator network analysis

## 📚 For Your Learning Hub

This tool is perfect for your n8nlearninghub.com content:

1. **Tutorial Series**: "Data-Driven n8n Workflow Design"
2. **Case Studies**: Analyze top workflows and explain why they work
3. **Trend Reports**: Monthly updates on n8n ecosystem
4. **Node Deep-Dives**: Focus on most popular nodes
5. **Creator Interviews**: Reach out to top creators

## 🤝 Contributing

This is a starting point! You can extend it to:
- Add more visualizations
- Implement NLP on descriptions
- Create interactive dashboards
- Build comparison tools
- Add export formats (CSV, Excel)

## 📄 License

Feel free to use this for your n8n Learning Hub content and tutorials!

---

**Happy Analyzing! 🎉**

For questions or improvements, reach out on the r/n8nLearningHub community!
