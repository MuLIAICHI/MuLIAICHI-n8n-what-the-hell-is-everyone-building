# n8n Workflow Analysis - Quick Reference Guide 🚀

## 📦 Files Overview

| File | Purpose |
|------|---------|
| `n8n_workflow_scraper.py` | Main scraper - collects workflow data |
| `n8n_workflow_analyzer.py` | Standard analysis - views, nodes, categories |
| `advanced_analysis.py` | Deep analysis - patterns, combinations, success factors |
| `run_all.py` | One-command solution - scrapes & analyzes |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

## ⚡ Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (scrape + analyze)
python run_all.py

# 3. Check results
ls -la n8n_data/analysis/
```

## 📊 What You'll Get

### Standard Analysis (`n8n_data/analysis/`)
- ✅ Top 20 most viewed workflows
- ✅ Top 30 most used nodes
- ✅ Top 15 categories/use cases
- ✅ Pricing analysis (free vs paid)
- ✅ Complexity distribution
- ✅ Top 15 creators
- ✅ 6 beautiful visualizations (PNG)
- ✅ Complete JSON report

### Advanced Analysis (`n8n_data/analysis_advanced/`)
- 🔥 Node combinations (which nodes work together)
- 🔥 Success factors (what makes workflows popular)
- 🔥 Workflow patterns (common structures)
- 🔥 Keyword analysis from descriptions
- 🔥 4 additional visualizations
- 🔥 Advanced JSON report

## 🎯 Common Use Cases

### 1. Quick Market Research
```bash
# Get all data in one go
python run_all.py
```

### 2. Focus on Specific Category
```python
# In Python console or script
from n8n_workflow_scraper import N8nWorkflowScraper

scraper = N8nWorkflowScraper()
workflows, filters = scraper.scrape_all_workflows(category='AI')
scraper.save_data(workflows, filters, prefix='ai_')
```

### 3. Get Just Top Workflows
```python
from n8n_workflow_analyzer import N8nWorkflowAnalyzer

analyzer = N8nWorkflowAnalyzer('n8n_data/raw/workflows_latest.json')
top_workflows = analyzer.analyze_top_viewed(top_n=50)
print(top_workflows)
```

### 4. Find Popular Node Combinations
```bash
# Run advanced analysis
python advanced_analysis.py
```

### 5. Update Your Data
```bash
# Re-scrape (takes 5-10 min)
python n8n_workflow_scraper.py

# Re-analyze
python n8n_workflow_analyzer.py
```

## 💡 Content Ideas for n8nlearninghub.com

### Beginner Tutorials
- "Start Here: The 5 Most Essential n8n Nodes"
- "Build Your First Workflow: 3-Node Starter"
- "Top 10 Free Workflows to Clone Today"

### Intermediate Tutorials
- "Master the HTTP Request Node" (most used!)
- "AI Agent Workflows: Complete Guide"
- "Google Sheets Automation: 20 Examples"

### Advanced Content
- "Node Combinations That Always Work"
- "Success Patterns: What Makes Workflows Popular"
- "Building Complex Workflows: 15+ Node Strategies"

### Market Analysis
- "n8n Trends 2025: What's Hot Right Now"
- "Most In-Demand Workflow Categories"
- "Pricing Strategies for n8n Templates"

### Creator Insights
- "Interview with [Top Creator Name]"
- "How Verified Creators Build Better Workflows"
- "Monetizing n8n Workflows: Real Numbers"

## 🔍 API Parameters

### Scraper Parameters
```python
scraper.scrape_all_workflows(
    category='AI',          # Filter by category (or None for all)
    rows_per_page=50,       # Results per page (1-100)
    max_pages=None          # Limit pages (None = get all)
)
```

### Available Categories
Based on your data, top categories are:
- AI, Multimodal AI, Marketing, AI Summarization
- Content Creation, AI Chatbot, Sales, Support
- AI RAG, and more...

## 📈 Data Structure

### Workflow Object
```json
{
  "id": 8429,
  "name": "Workflow Name",
  "totalViews": 1234,
  "price": 10,
  "purchaseUrl": "https://...",
  "user": {
    "id": 97839,
    "name": "Creator Name",
    "verified": true
  },
  "nodes": [
    {
      "id": 19,
      "name": "n8n-nodes-base.httpRequest",
      "displayName": "HTTP Request"
    }
  ]
}
```

## 🎨 Visualization Files

| File | Shows |
|------|-------|
| `top_viewed_workflows.png` | Most popular workflows bar chart |
| `top_used_nodes.png` | Node usage frequency |
| `top_categories.png` | Use case distribution |
| `pricing_analysis.png` | Free vs paid + price distribution |
| `complexity_analysis.png` | Node count distribution |
| `top_creators.png` | Most prolific creators |
| `node_combinations_heatmap.png` | Which nodes work together |
| `success_factors.png` | What makes workflows successful |
| `workflow_patterns.png` | Common starting/ending nodes |
| `description_keywords.png` | Most common terms used |

## 🐛 Troubleshooting

### "No module named 'requests'"
```bash
pip install -r requirements.txt
```

### "No data directory found"
```bash
# Run scraper first
python n8n_workflow_scraper.py
```

### Scraper timing out
```python
# Increase delay between requests
scraper.fetch_page(delay=3.0)  # Default is 1.5s
```

### Want fresh data
```bash
# Delete old data and re-scrape
rm -rf n8n_data/
python run_all.py
```

## 📊 Sample Output

### Top Nodes
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

### Top Categories
```
Category                       Usage Count
AI                                    4617
Multimodal AI                         2498
Marketing                             1717
```

## 🚀 Performance Tips

1. **For Speed**: Use `max_pages` parameter
   ```python
   workflows, _ = scraper.scrape_all_workflows(max_pages=10)
   ```

2. **For Completeness**: Scrape by category
   ```python
   workflows, _ = scraper.scrape_by_categories()
   ```

3. **For Updates**: Keep old data and compare
   ```bash
   # Scrape with timestamp prefix
   python n8n_workflow_scraper.py  # Auto-timestamps
   ```

## 💼 Business Use Cases

### For Agencies
- Identify trending workflow types
- Price your services competitively
- Find service gaps in market

### For Product Teams
- Understand user needs
- Prioritize feature development
- Find integration opportunities

### For Educators
- Create relevant curriculum
- Focus on high-demand skills
- Build practical examples

## 🤝 Next Steps

1. ✅ Run the scraper
2. ✅ Analyze the data
3. 📝 Create content based on insights
4. 🔄 Re-scrape monthly for trends
5. 📊 Share insights with community

## 📞 Support

- Reddit: r/n8nLearningHub
- Website: n8nlearninghub.com
- Issues: Check README.md for detailed troubleshooting

---

**Happy Analyzing! 🎉**

*Remember: Re-scrape periodically (weekly/monthly) to track trends!*
