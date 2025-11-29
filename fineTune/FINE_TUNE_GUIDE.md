# Fine-Tuning an LLM to Generate n8n Workflows

Complete guide to training an AI model that generates n8n workflows from natural language descriptions.

---

## 🎯 Goal

Train an LLM to understand requests like:
- "Create a chatbot that responds to Telegram messages using OpenAI"
- "Build a workflow that scrapes data from a website and saves it to Google Sheets"
- "Make an automation that monitors Gmail and sends alerts to Slack"

And output valid n8n workflow structures.

---

## 📊 Your Training Data

After running `prepare_training_data.py`, you'll have:
- **~6,000 training examples** (workflows with descriptions)
- **Multiple formats** (OpenAI, Alpaca, Simple)
- **Clean data** (no sensitive info, normalized descriptions)

Each training example pairs:
- **Input**: Natural language description
- **Output**: Workflow structure (nodes, types, relationships)

---

## 🛤️ Three Approaches

### Option 1: OpenAI Fine-Tuning (Easiest)
**Best for**: Quick results, production use  
**Cost**: ~$20-50 for fine-tuning, $0.012/1K tokens for inference  
**Time**: 1-2 hours

### Option 2: Open Source Models (Free)
**Best for**: Learning, no budget, full control  
**Cost**: $0 (uses your GPU or free Colab)  
**Time**: 4-8 hours

### Option 3: Local LoRA/QLoRA (Advanced)
**Best for**: Custom control, research, privacy  
**Cost**: $0 (needs decent GPU)  
**Time**: 8-24 hours

---

## 🚀 Option 1: OpenAI Fine-Tuning

### Step 1: Prepare Data

```bash
# Already done! Use the OpenAI format
python prepare_training_data.py
```

This creates: `n8n_data/training_data/training_data_openai.jsonl`

### Step 2: Upload and Fine-Tune

```python
import openai
from pathlib import Path

# Set your API key
openai.api_key = "sk-your-api-key-here"

# Upload training file
training_file = openai.File.create(
    file=open("n8n_data/training_data/training_data_openai.jsonl", "rb"),
    purpose='fine-tune'
)

print(f"Training file uploaded: {training_file.id}")

# Start fine-tuning (GPT-3.5-turbo recommended)
fine_tune = openai.FineTuningJob.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3  # Adjust based on data size
    }
)

print(f"Fine-tuning job created: {fine_tune.id}")
print("This will take 1-2 hours. Check status with:")
print(f"openai.FineTuningJob.retrieve('{fine_tune.id}')")
```

### Step 3: Use Your Model

```python
import openai

# After fine-tuning completes
response = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo:your-model-id",
    messages=[
        {"role": "system", "content": "You are an n8n workflow expert."},
        {"role": "user", "content": "Create a workflow that scrapes Hacker News and posts top stories to Slack"}
    ]
)

workflow = response.choices[0].message.content
print(workflow)
```

---

## 🔓 Option 2: Open Source Models

### Best Models for This Task:

1. **Llama 3 8B** (recommended for most)
2. **Mistral 7B** (great for structured output)
3. **Phi-3** (smaller, faster)
4. **CodeLlama 7B** (good with JSON)

### Step 1: Setup

```bash
# Install dependencies
pip install transformers datasets accelerate bitsandbytes peft

# Or use Google Colab (free GPU)
# https://colab.research.google.com
```

### Step 2: Fine-Tune with Hugging Face

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3-8b-hf"  # or "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load your training data
dataset = load_dataset('json', data_files='n8n_data/training_data/training_data_alpaca.json')

# Tokenize
def tokenize_function(examples):
    prompts = [f"### Instruction:\n{inst}\n\n### Response:\n{out}" 
               for inst, out in zip(examples['instruction'], examples['output'])]
    return tokenizer(prompts, truncation=True, padding="max_length", max_length=2048)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./n8n-workflow-generator",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

trainer.train()
trainer.save_model("./n8n-workflow-generator-final")
```

### Step 3: Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./n8n-workflow-generator-final")
tokenizer = AutoTokenizer.from_pretrained("./n8n-workflow-generator-final")

# Generate workflow
prompt = """### Instruction:
Create an n8n workflow for: AI-powered email responder

Description: Automatically respond to customer emails using GPT-4

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=1024, temperature=0.7)
workflow = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(workflow)
```

---

## ⚡ Option 3: LoRA Fine-Tuning (Most Efficient)

LoRA is faster and uses less memory than full fine-tuning.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train (same as before, but much faster!)
# ... training code ...

# Save LoRA adapters (tiny file, ~50MB vs 14GB full model)
model.save_pretrained("./n8n-lora-adapters")
```

---

## 🧪 Testing Your Model

```python
import json

def test_model(prompt: str):
    """Test your fine-tuned model"""
    # Generate workflow
    response = generate_workflow(prompt)  # Your inference function
    
    # Parse JSON
    try:
        workflow = json.loads(response)
        print("✓ Valid JSON structure")
        print(f"✓ Nodes: {len(workflow.get('nodes', []))}")
        print(f"✓ Node types: {workflow.get('node_types', [])}")
        return workflow
    except json.JSONDecodeError:
        print("✗ Invalid JSON - needs more training or better prompt")
        return None

# Test cases
test_prompts = [
    "Create a chatbot that uses OpenAI to respond to WhatsApp messages",
    "Build a workflow that scrapes Reddit and posts to Twitter",
    "Make an automation that monitors Gmail for invoices and saves them to Google Drive"
]

for prompt in test_prompts:
    print(f"\nTesting: {prompt}")
    workflow = test_model(prompt)
```

---

## 📈 Improving Model Performance

### 1. Data Quality
```python
# Filter workflows by quality metrics
def filter_high_quality_workflows(workflows, min_views=100, min_nodes=3):
    return [w for w in workflows 
            if w.get('totalViews', 0) >= min_views 
            and len(w.get('nodes', [])) >= min_nodes]
```

### 2. Data Augmentation
```python
# Create variations of descriptions
def augment_description(desc: str) -> List[str]:
    variations = [
        desc,
        f"Build a workflow that {desc.lower()}",
        f"Create an automation for {desc.lower()}",
        f"I need a workflow to {desc.lower()}"
    ]
    return variations
```

### 3. Prompt Engineering
```python
# Better system prompts
system_prompt = """You are an expert n8n workflow architect. 
Generate workflow structures that follow these principles:
1. Use standard n8n node types
2. Include proper node connections
3. Keep workflows simple and maintainable
4. Follow n8n best practices

Output valid JSON only."""
```

---

## 🎯 Evaluation Metrics

```python
def evaluate_model(test_set, model):
    """Evaluate model performance"""
    metrics = {
        'valid_json': 0,
        'correct_nodes': 0,
        'avg_nodes': [],
        'total': len(test_set)
    }
    
    for test in test_set:
        output = model.generate(test['input'])
        
        try:
            workflow = json.loads(output)
            metrics['valid_json'] += 1
            
            # Check if nodes match expected types
            if set(workflow['node_types']) == set(test['expected_nodes']):
                metrics['correct_nodes'] += 1
            
            metrics['avg_nodes'].append(len(workflow['nodes']))
        except:
            pass
    
    print(f"Valid JSON: {metrics['valid_json']}/{metrics['total']}")
    print(f"Correct nodes: {metrics['correct_nodes']}/{metrics['total']}")
    print(f"Avg nodes: {sum(metrics['avg_nodes'])/len(metrics['avg_nodes']):.1f}")
```

---

## 💡 Next Steps

1. **Start with OpenAI** if you want quick results
2. **Try Mistral 7B** for best free option
3. **Use LoRA** if you have GPU but limited VRAM
4. **Iterate on prompts** to improve output quality
5. **Build a web interface** (Gradio/Streamlit)
6. **Integrate with n8n API** to auto-create workflows

---

## 🚧 Challenges & Solutions

### Challenge 1: JSON Validation
**Solution**: Add JSON schema to training data, use strict output format

### Challenge 2: Hallucinated Nodes
**Solution**: Fine-tune on actual node types only, add validation layer

### Challenge 3: Complex Workflows
**Solution**: Start with simple workflows (5-10 nodes), expand gradually

### Challenge 4: Credentials/Secrets
**Solution**: Never include in training data, add placeholder system

---

## 📚 Resources

- **OpenAI Fine-tuning**: https://platform.openai.com/docs/guides/fine-tuning
- **Hugging Face**: https://huggingface.co/docs/transformers
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **n8n API Docs**: https://docs.n8n.io/api/

---

## 🎉 Example Output

**Input**: "Create a workflow that monitors GitHub issues and sends notifications to Slack"

**Output**:
```json
{
  "nodes": [
    {
      "name": "GitHub Trigger",
      "type": "GitHub Trigger",
      "category": ["Development"]
    },
    {
      "name": "Filter Issues",
      "type": "If",
      "category": ["Core Nodes"]
    },
    {
      "name": "Send to Slack",
      "type": "Slack",
      "category": ["Communication"]
    }
  ],
  "node_count": 3,
  "node_types": ["GitHub Trigger", "If", "Slack"]
}
```

---

**Ready to train?** Run `python prepare_training_data.py` and choose your approach!
