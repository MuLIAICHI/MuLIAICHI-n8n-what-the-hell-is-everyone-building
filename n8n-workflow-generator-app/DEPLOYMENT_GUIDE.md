# 🚀 Deployment Guide: n8n Workflow Generator Gradio App

Complete guide for deploying your n8n workflow generator to HuggingFace Spaces.

---

## 📋 Files You Need

```
n8n-workflow-generator-app/
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies
└── README.md              # HuggingFace Space README
```

---

## 🎯 Option 1: Deploy to HuggingFace Spaces (RECOMMENDED)

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   ```
   Name: n8n-workflow-generator-app
   License: MIT
   SDK: Gradio
   Hardware: CPU Basic (FREE) or T4 Small ($0.60/hour)
   ```
4. Click **"Create Space"**

### Step 2: Upload Files

**Method A: Via Web Interface**
1. Click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload all 3 files:
   - `app.py`
   - `requirements.txt` (rename from requirements_n8n.txt)
   - `README.md` (use README_SPACE.md)

**Method B: Via Git (Advanced)**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/n8n-workflow-generator-app
cd n8n-workflow-generator-app
cp app.py requirements_n8n.txt README_SPACE.md ./
mv requirements_n8n.txt requirements.txt
mv README_SPACE.md README.md
git add .
git commit -m "Initial deployment"
git push
```

### Step 3: Configure Space Settings

1. Go to **"Settings"** tab
2. Hardware: Select **"T4 Small"** for faster loading (recommended)
   - FREE tier works but model loads slowly (~2-3 minutes)
   - T4 Small: Loads in ~30 seconds
3. Save settings

### Step 4: Wait for Build

- Watch the **"Logs"** tab
- Build takes ~5-10 minutes first time
- Status will change to **"Running"** when ready

### Step 5: Test Your App

1. Click your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/n8n-workflow-generator-app`
2. Click **"Load Model"** button
3. Try an example: "Build a Telegram chatbot that uses OpenAI"
4. Click **"Generate Workflow"**

---

## 🖥️ Option 2: Run Locally (For Testing)

### Requirements

- Python 3.10+
- CUDA GPU (recommended) or CPU (slower)
- 8GB+ RAM
- 20GB disk space for model

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install gradio torch transformers accelerate sentencepiece protobuf

# 3. Run the app
python app.py
```

### Access

Open browser to: `http://localhost:7860`

---

## 🎛️ Option 3: Deploy to Railway

### Prerequisites

- Railway account (https://railway.app)
- GitHub account

### Steps

1. **Push to GitHub**:
   ```bash
   git init
   git add app.py requirements_n8n.txt README_SPACE.md
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO
   git push -u origin main
   ```

2. **Deploy on Railway**:
   - Go to https://railway.app
   - Click **"New Project"** → **"Deploy from GitHub repo"**
   - Select your repository
   - Railway auto-detects Python
   - Set port to `7860`

3. **Configure**:
   - Add environment variable: `PORT=7860`
   - Allocate at least 8GB RAM

---

## 🐛 Troubleshooting

### Issue: Model Not Loading

**Symptoms**: Error when clicking "Load Model"

**Solutions**:
1. Check HuggingFace is accessible
2. Verify model exists: https://huggingface.co/MustaphaL/n8n-workflow-generator
3. Check Space logs for errors
4. Try upgrading to T4 Small hardware

### Issue: Out of Memory

**Symptoms**: Crash during model loading

**Solutions**:
1. Upgrade to T4 Small (16GB RAM)
2. Or use CPU Basic with patience (slow but works)

### Issue: Slow Generation

**Normal behavior**:
- CPU Basic: 30-60 seconds per generation
- T4 Small: 5-10 seconds per generation

**If slower than this**:
1. Check hardware tier
2. Restart Space
3. Check logs for errors

### Issue: Invalid JSON Output

**Cause**: Model sometimes generates explanatory text with JSON

**Solution**: Extract JSON from raw output manually, or adjust temperature lower (0.5-0.6)

---

## 📊 Hardware Recommendations

| Hardware | Load Time | Generation Time | Cost | Recommended For |
|----------|-----------|----------------|------|-----------------|
| CPU Basic | 2-3 min | 30-60 sec | FREE | Testing, demos |
| T4 Small | 30 sec | 5-10 sec | $0.60/hr | Production, frequent use |
| T4 Medium | 20 sec | 3-5 sec | $1.20/hr | High traffic |

---

## 🎨 Customization Options

### Change Theme

In `app.py`, line 166:
```python
gr.Blocks(css=custom_css, theme=gr.themes.Soft())
```

Options: `Soft()`, `Monochrome()`, `Glass()`, `Base()`

### Adjust Model Parameters

Default values in app:
- Max tokens: 512
- Temperature: 0.7
- Top P: 0.9

Modify sliders in `app.py` lines 180-210.

### Add More Examples

Edit `examples` list in `app.py` line 120:
```python
examples = [
    ["Your new example here"],
    # ... more examples
]
```

---

## 📈 Monitoring

### Check Space Stats

1. Go to Space settings
2. View **"Analytics"** tab
3. Monitor:
   - API calls
   - Average response time
   - Error rate

### Enable Auto-Sleep

To save costs:
1. Settings → **"Sleep time"**
2. Set to **"1 hour"** (Space sleeps after inactivity)
3. Wakes up automatically on access

---

## 🔐 Security Notes

- Model is public by default
- No API keys needed for this app
- Generated workflows are not stored
- All processing happens in user's browser session

---

## 💰 Cost Estimates

**FREE Tier (CPU Basic)**:
- ✅ Completely free
- ❌ Slower performance
- ✅ Perfect for personal use

**Paid Tier (T4 Small - $0.60/hour)**:
- Monthly cost if running 24/7: ~$432
- With auto-sleep (2hr/day active): ~$36/month
- With manual control (4hr/week): ~$10/month

**Recommendation**: Start with FREE, upgrade if you get traction!

---

## 🎉 You're Done!

Your n8n Workflow Generator is now live!

**Next Steps:**
1. Share your Space URL on social media
2. Write Medium article about deployment (Part 3!)
3. Post on r/n8nLearningHub
4. Add to your website: n8nlearninghub.com

---

## 🆘 Need Help?

- 💬 Reddit: r/n8nLearningHub
- 🤗 HuggingFace: Leave comment on model page
- 📧 Email: [your email]

---

**Built with ❤️ by MHL | Powered by Gradio + Llama 3 8B**
