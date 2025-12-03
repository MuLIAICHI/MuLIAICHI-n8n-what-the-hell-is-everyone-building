# n8n Marketplace Intelligence Suite 🚀

**From curiosity to production:** A complete journey of scraping 6,000+ n8n workflows, analyzing patterns, fine-tuning an LLM, and deploying a web app that generates workflows from natural language.

[![Try the App](https://img.shields.io/badge/🚀-Try_Live_App-orange)](https://huggingface.co/spaces/MustaphaL/n8n-workflow-generator-app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuLIAICHI/n8n-marketplace-analyzer/blob/main/fineTune/n8n_workflow_generator_fine_tuning.ipynb)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/MustaphaL/n8n-workflow-generator)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data)
[![Apify Actor](https://img.shields.io/badge/Apify-Actor-success)](https://apify.com/scraper_guru/n8n-marketplace-analyzer)

---

## 📖 The Complete Journey

This project started with a simple question: **"What are people actually building with n8n?"**

It evolved into:
1. **Scraping & Analysis** - 6,837 workflows analyzed
2. **Production Tool** - Apify actor with analytics
3. **ML Training Data** - 4,000+ examples generated
4. **LLM Fine-tuning** - Llama 3 8B trained to generate workflows
5. **Web Application** - Gradio app deployed on HuggingFace Spaces
6. **Public Resources** - Everything open-sourced

**📝 Read the complete story:**
- [Part 1: What Are People Actually Building in n8n?](https://medium.com/@mustaphaliaichi/what-are-people-actually-building-in-n8n-i-scraped-over-6-000-workflows-to-find-out-59eb8e34c317) - 61K Reddit views
- [Part 2: I Fine-Tuned Llama 3 on 6,000 n8n Workflows](https://medium.com/@mustaphaliaichi/i-fine-tuned-llama-3-on-6-000-n8n-workflows-after-mistral-failed-spectacularly-927cce57df14) - Fine-tuning journey from failure to success
- [Part 3: From Curiosity to Deployment](https://medium.com/@mustaphaliaichi/from-curiosity-to-deployment-how-i-turned-6-000-n8n-workflows-into-an-ai-generator-and-why-this-e916923826ea) - Building and deploying the Gradio app

---

## 🎯 Project Components

### 1. Data Collection & Analysis 📊

**Scripts:**
- `n8n_workflow_scraper.py` - Scrapes n8n marketplace
- `n8n_workflow_analyzer.py` - Analyzes patterns and trends
- `advanced_analysis.py` - Deep-dive analytics

**What it does:**
- ✅ Scrapes 6,000+ workflows from n8n marketplace
- ✅ Smart pagination & rate limiting
- ✅ Comprehensive analysis (nodes, categories, pricing)
- ✅ Beautiful visualizations
- ✅ Trend discovery

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Scrape marketplace
python n8n_workflow_scraper.py

# Analyze data
python n8n_workflow_analyzer.py
```

**Key Findings:**
- 73% of workflows use HTTP Request node
- 32% integrate AI (OpenAI, Claude, LangChain)
- 86% of workflows are free
- Average workflow: 8.5 nodes
- 40% are simple (1-5 nodes)

---

### 2. Production Tool (Apify Actor) 🛠️

**Live Tool:** [n8n Marketplace Analyzer](https://apify.com/scraper_guru/n8n-marketplace-analyzer)

**Features:**
- 🔍 Scrape 1-10,000 workflows on-demand
- 📊 Comprehensive analytics (top nodes, categories, pricing)
- 🧠 ML training data generation (Alpaca & OpenAI formats)
- 💰 Currently free (Apify $1M Challenge participant)

**What makes it unique:**
- Goes beyond basic scraping
- Provides actionable insights
- Generates ML training datasets
- Production-ready infrastructure

---

### 3. ML Training Dataset 📚

**HuggingFace Dataset:** [n8n-workflow-training-data](https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data)

**Contents:**
- **4,000+ training examples** extracted from 6,837 workflows
- **Three formats:**
  - `training_data_alpaca.json` - For Llama/Mistral
  - `training_data_openai.jsonl` - For GPT
  - `training_data_simple.json` - For custom pipelines

**Sample Format:**
```json
{
  "instruction": "Create an n8n workflow for: AI Email Assistant",
  "input": "",
  "output": {
    "name": "AI Email Assistant",
    "nodes": [
      {"type": "Gmail Trigger"},
      {"type": "OpenAI Chat Model"},
      {"type": "Gmail"}
    ],
    "node_count": 3,
    "categories": ["AI", "Communication"]
  }
}
```

**Use Case:** Fine-tune LLMs to generate n8n workflows from natural language

---

### 4. Fine-Tuned LLM Model 🤖

**HuggingFace Model:** [n8n-workflow-generator](https://huggingface.co/MustaphaL/n8n-workflow-generator)

**Model Details:**
- **Base:** Llama 3 8B (4-bit quantized)
- **Training:** 1,283 curated examples
- **Time:** 55 minutes on A100
- **Final Loss:** 1.235900
- **Quality:** Production-ready

**What it does:**
Generate n8n workflow configurations from natural language descriptions.

**Example:**
```python
Input: "Build a Telegram chatbot that uses OpenAI"

Output:
{
  "nodes": [
    {"type": "Telegram Trigger"},
    {"type": "OpenAI Chat Model"},
    {"type": "Telegram"}
  ],
  "node_count": 3
}
```

**Training Notebook:** [Open in Colab](https://colab.research.google.com/github/MuLIAICHI/n8n-marketplace-analyzer/blob/main/fineTune/n8n_workflow_generator_fine_tuning.ipynb)

**The Journey:**
- ❌ First attempt: Mistral 7B (catastrophic overfitting - loss → 0, gibberish output)
- ✅ Second attempt: Llama 3 8B (smooth training, valid outputs)
- 📖 Full story in [Part 2 article](https://medium.com/@mustaphaliaichi/i-fine-tuned-llama-3-on-6-000-n8n-workflows-after-mistral-failed-spectacularly-927cce57df14)

---

### 5. Web Application (Gradio App) 🌐

**Live App:** [Try it now!](https://huggingface.co/spaces/MustaphaL/n8n-workflow-generator-app)

**What it does:**
- ✅ Generate n8n workflows from plain English descriptions
- ✅ No coding required - just describe what you want
- ✅ Instant JSON output ready to import into n8n
- ✅ Advanced settings (temperature, max tokens)
- ✅ Example prompts included
- ✅ Free to use (hosted on HuggingFace Spaces)

**Features:**
- **Simple Interface:** Just type what you want to automate
- **Smart Loading:** Model loads once, stays in memory
- **JSON Extraction:** Automatically extracts clean workflow JSON
- **Dual Output:** See both raw generation and formatted JSON
- **Example Prompts:** 8 built-in examples to get started

**How to Use:**
1. Go to the [live app](https://huggingface.co/spaces/MustaphaL/n8n-workflow-generator-app)
2. Click "Load Model" (wait ~30 seconds)
3. Describe your workflow in plain English
4. Click "Generate Workflow"
5. Copy the JSON and import into n8n

**Example Prompts:**
- "Build a Telegram chatbot that uses OpenAI to respond to messages"
- "Create a workflow that monitors Gmail and sends Slack notifications"
- "Build an automation that scrapes product prices and saves to Google Sheets"

**Read the deployment story:** [Part 3 article](https://medium.com/@mustaphaliaichi/from-curiosity-to-deployment-how-i-turned-6-000-n8n-workflows-into-an-ai-generator-and-why-this-e916923826ea)

---

## 📂 Repository Structure

```
n8n-marketplace-analyzer/
├── fineTune/                              # LLM Fine-tuning
│   ├── n8n_workflow_generator_fine_tuning.ipynb  # Colab notebook
│   ├── COMPARISON_GUIDE.md                # Mistral vs Llama comparison
│   ├── FINE_TUNE_GUIDE.md                # Fine-tuning tutorial
│   ├── QUICK_START_FINETUNING.md         # Quick start guide
│   └── UNSLOTH_GUIDE.md                  # Unsloth library guide
├── n8n_data/                              # Data storage
│   ├── raw/                               # Scraped workflows
│   ├── analysis/                          # Analysis results
│   ├── processed/                         # Processed data
│   └── training_data/                     # ML training datasets
├── n8n_workflow_scraper.py                # Main scraper
├── n8n_workflow_analyzer.py               # Main analyzer
├── advanced_analysis.py                   # Deep analytics
├── run_all.py                             # Run complete pipeline
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## 🚀 Quick Start

### Option 1: Use the Web App (Easiest!)

**Try the live app:**
```
1. Go to: https://huggingface.co/spaces/MustaphaL/n8n-workflow-generator-app
2. Click "Load Model" (wait 30 seconds)
3. Describe your workflow in plain English
4. Click "Generate Workflow"
5. Copy JSON and import into n8n
```

No installation. No coding. Just works.

### Option 2: Use Pre-Built Resources

**Try the Apify Actor:**
```
1. Go to: https://apify.com/scraper_guru/n8n-marketplace-analyzer
2. Click "Try for free"
3. Enter parameters (1-10,000 workflows)
4. Get comprehensive analytics + ML data
```

**Use the Fine-Tuned Model:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "MustaphaL/n8n-workflow-generator",
    max_seq_length = 2048,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

# Generate workflow
inputs = tokenizer(
    "Create an n8n workflow for: Email automation with Google Drive",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

**Download Training Data:**
```bash
# From HuggingFace
git clone https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data
```

### Option 3: Run Locally

**1. Clone Repository:**
```bash
git clone https://github.com/MuLIAICHI/n8n-marketplace-analyzer.git
cd n8n-marketplace-analyzer
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Scrape & Analyze:**
```bash
# Complete pipeline
python run_all.py

# Or run individually
python n8n_workflow_scraper.py
python n8n_workflow_analyzer.py
```

### Option 4: Fine-Tune Your Own Model

**Open in Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuLIAICHI/n8n-marketplace-analyzer/blob/main/fineTune/n8n_workflow_generator_fine_tuning.ipynb)

**Requirements:**
- Google Colab Pro ($9.99/month) for A100 GPU
- ~55 minutes training time
- Follow notebook instructions

---

## 📊 Analysis Capabilities

### What You Can Discover:

**1. Popularity Metrics**
- Most viewed workflows
- View count distributions
- Trending patterns

**2. Node Usage Analysis**
- Top 30 most-used nodes
- Essential vs specialized nodes
- Node combinations

**3. Use Case Distribution**
- Top 15 categories
- Real-world applications
- Emerging trends (AI dominating!)

**4. Pricing Insights**
- Free vs paid distribution
- Average prices
- Pricing strategies

**5. Complexity Analysis**
- Simple vs complex workflows
- Node count distributions
- Optimal complexity ranges

**6. Creator Analytics**
- Most prolific creators
- Verified creators
- Quality indicators

---

## 📈 Key Insights (From 6,837 Workflows)

### Top Nodes
1. **HTTP Request** - 73% (API integration is king!)
2. **Code** - 52% (JavaScript still essential)
3. **Set** - 48% (Data transformation)
4. **IF** - 45% (Conditional logic)
5. **OpenAI** - 32% (AI revolution!)

### Categories
1. **AI** - 4,617 workflows
2. **LangChain** - 2,498 workflows
3. **Development** - 1,717 workflows
4. **Marketing** - 1,294 workflows
5. **Content Creation** - 1,081 workflows

### Pricing
- **86% FREE** - Community is generous!
- **14% PAID** - Average $12.50
- Range: $0 to $99

### Complexity
- **Simple (1-5 nodes):** 40%
- **Medium (6-10 nodes):** 35%
- **Complex (11-20 nodes):** 18%
- **Very Complex (20+ nodes):** 7%

---

## 🎓 Educational Resources

### Guides (in `/fineTune/`)
- **COMPARISON_GUIDE.md** - Mistral vs Llama comparison
- **FINE_TUNE_GUIDE.md** - Complete fine-tuning tutorial
- **QUICK_START_FINETUNING.md** - Get started quickly
- **UNSLOTH_GUIDE.md** - Efficient fine-tuning with Unsloth

### Blog Series (Complete!)
1. [Part 1: Scraping & Analysis](https://medium.com/@mustaphaliaichi/what-are-people-actually-building-in-n8n-i-scraped-over-6-000-workflows-to-find-out-59eb8e34c317)
   - The question, the scraping journey, key findings
   - 61K Reddit views, 181 upvotes
   
2. [Part 2: Fine-Tuning Journey](https://medium.com/@mustaphaliaichi/i-fine-tuned-llama-3-on-6-000-n8n-workflows-after-mistral-failed-spectacularly-927cce57df14)
   - From Mistral failure to Llama success
   - Technical deep-dive, lessons learned
   
3. [Part 3: Deployment](https://medium.com/@mustaphaliaichi/from-curiosity-to-deployment-how-i-turned-6-000-n8n-workflows-into-an-ai-generator-and-why-this-e916923826ea)
   - Building the Gradio app
   - Deployment to HuggingFace Spaces
   - Why this is where the journey ends

---

## 💻 Use Cases

### For n8n Users
- **Generate Workflows:** Use the web app to create workflows from descriptions
- **Learn Patterns:** See what successful workflows look like
- **Get Started Faster:** Jump-start your automation projects

### For Content Creators
- **Content Ideas** - See what topics get views
- **Tutorial Planning** - Focus on popular nodes
- **Market Gaps** - Find underserved categories
- **SEO Keywords** - Popular search terms

### For Developers
- **Learning Path** - Start with common nodes
- **Best Practices** - Study popular workflows
- **Integration Ideas** - Popular tool combinations
- **Quality Signals** - Identify patterns in successful workflows

### For Researchers/ML Engineers
- **Training Data** - Ready-to-use datasets
- **Fine-Tuning** - Pre-trained model available
- **Pattern Analysis** - Real-world workflow patterns
- **Use Case Discovery** - Automation trends

### For Business Analysis
- **Market Research** - n8n ecosystem insights
- **Product Strategy** - Feature priorities
- **Competitive Analysis** - Compare offerings
- **Trend Forecasting** - Emerging use cases

---

## 🛠️ Tech Stack

**Data Collection:**
- Python 3.12
- Requests (API calls)
- Pandas (data processing)
- Matplotlib/Seaborn (visualizations)

**Production Tool:**
- Apify SDK
- Docker
- Cloud infrastructure

**ML Pipeline:**
- Unsloth (efficient fine-tuning)
- Transformers (Hugging Face)
- PEFT (LoRA fine-tuning)
- Weights & Biases (tracking)

**Web Application:**
- Gradio 4.44.0
- HuggingFace Spaces
- PyTorch & Transformers

**Hardware:**
- Google Colab Pro (A100 GPU)
- ~$10/month for training

---

## 📊 Training Results

### Mistral 7B (Failed ❌)
- **Initial Loss:** 8.135
- **Final Loss:** 0.0001 (by step 50!)
- **Result:** Catastrophic overfitting
- **Output:** Gibberish (トトトトトト...)
- **Attempts:** 5 different configurations
- **Lesson:** Model selection matters!

### Llama 3 8B (Success ✅)
- **Initial Loss:** 1.293
- **Final Loss:** 1.235900
- **Training Time:** 55 min 46 sec
- **GPU:** A100 (Colab Pro)
- **Quality:** 15/15 test prompts
- **Result:** Production-ready!
- **Deployed:** Live web app serving users

---

## 🌐 All Resources

**🚀 Live Applications:**
- **Web App:** [n8n-workflow-generator-app](https://huggingface.co/spaces/MustaphaL/n8n-workflow-generator-app)
- **Apify Actor:** [n8n-marketplace-analyzer](https://apify.com/scraper_guru/n8n-marketplace-analyzer)

**🤗 Models & Data:**
- **Model:** [MustaphaL/n8n-workflow-generator](https://huggingface.co/MustaphaL/n8n-workflow-generator)
- **Dataset:** [MustaphaL/n8n-workflow-training-data](https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data)

**🛠️ Tools:**
- **Colab Notebook:** [Fine-tuning Tutorial](https://colab.research.google.com/github/MuLIAICHI/n8n-marketplace-analyzer/blob/main/fineTune/n8n_workflow_generator_fine_tuning.ipynb)

**📝 Complete Article Series:**
- **Part 1:** [What Are People Actually Building?](https://medium.com/@mustaphaliaichi/what-are-people-actually-building-in-n8n-i-scraped-over-6-000-workflows-to-find-out-59eb8e34c317)
- **Part 2:** [Fine-Tuning After Mistral Failed](https://medium.com/@mustaphaliaichi/i-fine-tuned-llama-3-on-6-000-n8n-workflows-after-mistral-failed-spectacularly-927cce57df14)
- **Part 3:** [From Curiosity to Deployment](https://medium.com/@mustaphaliaichi/from-curiosity-to-deployment-how-i-turned-6-000-n8n-workflows-into-an-ai-generator-and-why-this-e916923826ea)

**🌐 Community:**
- **Website:** [n8nlearninghub.com](https://n8nlearninghub.com)
- **Reddit:** [automata_n8n](https://www.reddit.com/user/automata_n8n/)
- **GitHub:** [@MuLIAICHI](https://github.com/MuLIAICHI)
- **Medium:** [@mustaphaliaichi](https://medium.com/@mustaphaliaichi)

---

## 🎯 What Makes This Project Unique

### 1. Complete Journey
Not just a scraper - it's the full story from curiosity to production.

### 2. Real Value
- Free web app for generating workflows
- Free analytics tool (Apify)
- Open training data
- Pre-trained model
- Educational content

### 3. Build in Public
- Shared failures (Mistral overfitting)
- Shared successes (Llama working)
- Complete transparency
- Full documentation

### 4. Community First
- 61K article views
- 1,000+ community members
- Everything open-source
- Free resources

### 5. Production Quality
- Not a toy project
- Real users using the web app
- Maintained code
- Documented thoroughly
- Actually finished!

---

## 📚 Citation

If you use this project, model, dataset, or app in your work:

```bibtex
@software{liaichi2024n8nanalyzer,
  author = {Mustapha Liaichi},
  title = {n8n Marketplace Intelligence Suite},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MuLIAICHI/n8n-marketplace-analyzer}
}

@model{liaichi2024n8nmodel,
  author = {Mustapha Liaichi},
  title = {n8n Workflow Generator (Llama 3 8B Fine-tuned)},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/MustaphaL/n8n-workflow-generator}
}

@dataset{liaichi2024n8ndata,
  author = {Mustapha Liaichi},
  title = {n8n Workflow Training Dataset},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data}
}

@software{liaichi2024n8napp,
  author = {Mustapha Liaichi},
  title = {n8n Workflow Generator Web App},
  year = {2024},
  publisher = {Hugging Face Spaces},
  url = {https://huggingface.co/spaces/MustaphaL/n8n-workflow-generator-app}
}
```

---

## 🤝 Contributing

This is a starting point! You can extend it to:
- Add more visualizations
- Implement NLP on descriptions
- Create interactive dashboards
- Build comparison tools
- Add export formats (CSV, Excel)
- Improve the web app UI
- Add more training data

## 📄 License

MIT License - Feel free to use this for your n8n Learning Hub content and tutorials!

---

**Happy Automating! 🎉**

For questions or improvements, reach out at mustaphaliaichi@gmail.com

**Built with curiosity, completed with persistence.**