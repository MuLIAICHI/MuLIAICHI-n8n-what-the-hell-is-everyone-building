# Fine-Tuning Quick Start Guide

Transform 6,000+ n8n workflows into an AI that generates workflows from natural language!

---

## 🎯 What You're Building

An LLM that can:
- Understand: "Create a chatbot that responds to WhatsApp messages using OpenAI"
- Output: Valid n8n workflow structure with proper nodes

---

## ⚡ Quick Start (5 Steps)

### Step 1: Prepare Training Data (5 minutes)

```bash
# Make sure you've scraped the workflows first
python n8n_workflow_scraper.py  # If you haven't already

# Convert workflows to training data
python prepare_training_data.py
```

**Output:**
- `training_data_openai.jsonl` - For OpenAI fine-tuning
- `training_data_alpaca.json` - For open source models
- `training_data_simple.json` - For custom approaches

You'll get ~6,000 training examples!

---

### Step 2: Choose Your Path

#### 🚀 Path A: OpenAI (Easiest, $20-50)

```bash
# Install OpenAI
pip install openai

# Upload and fine-tune
python -c "
import openai
openai.api_key = 'sk-your-key'

# Upload training file
file = openai.File.create(
    file=open('n8n_data/training_data/training_data_openai.jsonl', 'rb'),
    purpose='fine-tune'
)

# Start fine-tuning
job = openai.FineTuningJob.create(
    training_file=file.id,
    model='gpt-3.5-turbo'
)

print(f'Job ID: {job.id}')
print('Check status at: https://platform.openai.com/finetune')
"
```

**Time:** 1-2 hours  
**Cost:** ~$20-50  
**Result:** Production-ready model

---

#### 🔓 Path B: Open Source (Free, 4-8 hours)

```bash
# Install dependencies
pip install -r requirements_finetuning.txt

# Use Google Colab for free GPU
# https://colab.research.google.com

# Upload training_data_alpaca.json and run:
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load Mistral 7B (recommended)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load your data
dataset = load_dataset('json', data_files='training_data_alpaca.json')

# Tokenize
def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return tokenizer(text, truncation=True, max_length=2048)

dataset = dataset.map(tokenize)

# Train
trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    args=TrainingArguments(
        output_dir="./n8n-generator",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        fp16=True,
    )
)

trainer.train()
trainer.save_model("./n8n-generator-final")
```

**Time:** 4-8 hours on free Colab GPU  
**Cost:** $0  
**Result:** Fully owned model

---

### Step 3: Generate Workflows

#### Using OpenAI Model:

```bash
python generate_workflow.py \
    --model-type openai \
    --model-id ft:gpt-3.5-turbo:your-org:n8n:abc123 \
    --description "Create a workflow that scrapes Hacker News and posts to Slack"
```

#### Using Your Fine-Tuned Model:

```bash
python generate_workflow.py \
    --model-type huggingface \
    --model-path ./n8n-generator-final \
    --description "Build a chatbot that answers questions using OpenAI"
```

---

### Step 4: Test & Iterate

```bash
# Create test cases
cat > test_descriptions.txt << EOF
Create a Telegram bot that responds using ChatGPT
Build a workflow that monitors Gmail and saves attachments to Google Drive
Make an automation that scrapes a website daily and sends updates via email
Generate social media posts using AI and schedule them
EOF

# Batch test
python generate_workflow.py \
    --model-type openai \
    --model-id your-model-id \
    --batch test_descriptions.txt \
    --output results.json
```

---

### Step 5: Deploy (Optional)

#### Create a Simple API:

```python
# api.py
from fastapi import FastAPI
from generate_workflow import N8nWorkflowGenerator

app = FastAPI()
generator = N8nWorkflowGenerator(model_type="openai")

@app.post("/generate")
async def generate(description: str):
    result = generator.generate(description)
    return result

# Run: uvicorn api:app --reload
```

#### Or Build a UI with Gradio:

```python
# ui.py
import gradio as gr
from generate_workflow import N8nWorkflowGenerator

generator = N8nWorkflowGenerator(model_type="openai")

def generate_ui(description):
    result = generator.generate(description)
    if result['success']:
        return result['workflow']
    return {"error": result['error']}

interface = gr.Interface(
    fn=generate_ui,
    inputs=gr.Textbox(label="Describe your workflow"),
    outputs=gr.JSON(label="Generated Workflow"),
    title="n8n Workflow Generator"
)

interface.launch()
```

---

## 📊 Expected Results

### Training Data Stats:
- **~6,000 examples** from marketplace
- **Average description:** 200-300 characters
- **Average workflow:** 8 nodes
- **Most common nodes:** HTTP Request, Code, AI Agent

### Model Performance:
- **Valid JSON:** 85-95% (after fine-tuning)
- **Correct node types:** 70-80%
- **Usable workflows:** 60-70%

### Improvement Tips:
1. **Filter by quality:** Use workflows with 100+ views
2. **Add examples:** Include your own custom workflows
3. **Prompt engineering:** Improve system prompts
4. **Validation layer:** Add JSON schema validation

---

## 🎓 Learning Path

**Beginner:**
1. Start with OpenAI fine-tuning (easiest)
2. Test with simple workflows (3-5 nodes)
3. Iterate on prompts

**Intermediate:**
1. Try Mistral 7B on Colab
2. Experiment with LoRA
3. Add custom validation

**Advanced:**
1. Train locally with your own data
2. Create ensemble models
3. Build production API
4. Integrate with n8n directly

---

## 🔥 Pro Tips

1. **Start small:** Test with 100-500 examples first
2. **Quality over quantity:** Filter high-view workflows
3. **Validate output:** Always check generated JSON
4. **Iterate fast:** Don't aim for perfect on first try
5. **Use templates:** Common patterns (chatbot, scraper, etc.)

---

## 📁 File Structure

```
your-project/
├── n8n_data/
│   ├── raw/
│   │   └── workflows_*.json         # Scraped data
│   └── training_data/
│       ├── training_data_openai.jsonl
│       ├── training_data_alpaca.json
│       └── training_data_simple.json
├── prepare_training_data.py          # Convert workflows
├── generate_workflow.py              # Inference script
├── requirements_finetuning.txt       # Dependencies
├── FINE_TUNE_GUIDE.md               # Full guide
└── QUICK_START.md                   # This file
```

---

## 🚀 Next Steps

After fine-tuning works:

1. **Build a UI:** Gradio or Streamlit
2. **Add validation:** Check node compatibility
3. **Create templates:** For common patterns
4. **Integrate with n8n:** Auto-create via API
5. **Share with community:** Open source it!

---

## 🆘 Troubleshooting

### "Out of memory"
→ Use smaller batch size or LoRA/QLoRA

### "Invalid JSON output"
→ Add more training examples with valid JSON

### "Wrong node types"
→ Filter training data to include only real nodes

### "Slow inference"
→ Use quantization or smaller model

---

## 📚 Resources

- **Full Guide:** `FINE_TUNE_GUIDE.md`
- **OpenAI Docs:** https://platform.openai.com/docs/guides/fine-tuning
- **Hugging Face:** https://huggingface.co/docs/transformers
- **n8n Community:** https://community.n8n.io

---

## 🎉 Success Criteria

Your model is working when:
- ✅ Generates valid JSON 90%+ of the time
- ✅ Uses correct node types for task
- ✅ Creates logical node connections
- ✅ Handles various workflow types

---

**Ready? Run:** `python prepare_training_data.py`

Then choose Path A (OpenAI) or Path B (Open Source) and start training! 🚀
