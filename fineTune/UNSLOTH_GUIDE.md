# Fine-Tuning n8n Workflow Generator with Unsloth

Use Unsloth to train your LLM **2-5x faster** with **70% less memory**. Perfect for n8n workflow generation!

## 🚀 Why Unsloth?

- ⚡ **2-5x faster** than standard fine-tuning
- 💾 **70% less memory** - train bigger models on free Colab
- 🎯 **Easy to use** - just a few lines of code
- 🔓 **Free & open source**
- 🤝 **Works with Hugging Face models**

---

## 📦 Quick Setup

### Option 1: Google Colab (Free GPU - Recommended!)

```bash
# In a Colab notebook
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Option 2: Local Installation

```bash
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
# or for CUDA 11.8: pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

---

## 🎯 Complete Training Script

```python
# n8n_workflow_finetune_unsloth.py

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. Load Model (choose one)
max_seq_length = 2048  # Supports up to 8192 for Llama-3
dtype = None  # Auto-detect. Set to Float16 for faster training
load_in_4bit = True  # Use 4bit quantization to reduce memory

# Supported models (pick one):
# - "unsloth/llama-3-8b-bnb-4bit"
# - "unsloth/mistral-7b-v0.3-bnb-4bit"
# - "unsloth/phi-3-mini-4k-instruct-bnb-4bit"
# - "unsloth/gemma-2-9b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit",  # Best for this task
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank (higher = more parameters, slower)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,  # Supports any, but = 0 is optimized
    bias = "none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",  # Very long contexts
    random_state = 3407,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None,  # LoftQ
)

# 3. Load Your Training Data
dataset = load_dataset('json', data_files='n8n_data/training_data/training_data_alpaca.json')

# 4. Format Data for Training
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Training Configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # Can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,  # Adjust based on your data
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",  # Use "wandb" for tracking
    ),
)

# 6. Train!
print("🚀 Starting training...")
trainer_stats = trainer.train()

# 7. Save Model
print("💾 Saving model...")
model.save_pretrained("n8n_workflow_generator_unsloth")
tokenizer.save_pretrained("n8n_workflow_generator_unsloth")

print("✅ Training complete!")
print(f"Time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
```

---

## ⚡ Google Colab Notebook (Copy-Paste Ready)

```python
# === CELL 1: Install ===
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# === CELL 2: Upload Data ===
from google.colab import files
print("Upload your training_data_alpaca.json file:")
uploaded = files.upload()

# === CELL 3: Train ===
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Load data
dataset = load_dataset('json', data_files='training_data_alpaca.json')

# Format data
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("🚀 Starting training...")
trainer_stats = trainer.train()

# Save
model.save_pretrained("n8n_generator")
tokenizer.save_pretrained("n8n_generator")

print("✅ Done! Training time:", trainer_stats.metrics['train_runtime'], "seconds")

# === CELL 4: Test ===
# Enable fast inference
FastLanguageModel.for_inference(model)

# Test prompt
instruction = "Create an n8n workflow for: Build a Telegram chatbot that uses OpenAI to respond to messages"

inputs = tokenizer(
    [alpaca_prompt.format(instruction, "")],
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens = 512,
    temperature = 0.7,
    top_p = 0.9,
    use_cache = True
)

result = tokenizer.batch_decode(outputs)
print(result[0])

# === CELL 5: Download Model ===
from google.colab import files

# Zip the model
!zip -r n8n_generator.zip n8n_generator/

# Download
files.download('n8n_generator.zip')
```

---

## 🎯 Inference After Training

```python
# load_and_generate.py

from unsloth import FastLanguageModel

# Load your trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "n8n_generator",  # Your saved model
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Generate workflow
def generate_workflow(description: str):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create an n8n workflow for: {description}

### Response:
"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,
        temperature = 0.7,
        top_p = 0.9,
        use_cache = True
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response
    response = result.split("### Response:")[-1].strip()
    return response

# Test
workflow = generate_workflow("Build a chatbot that monitors Gmail and responds via Slack")
print(workflow)
```

---

## 📊 Expected Performance

### Training Speed (Free Colab T4 GPU):
- **Without Unsloth**: ~6-8 hours for 6,000 examples
- **With Unsloth**: ~2-3 hours for 6,000 examples
- **Speedup**: 2.5-3x faster! ⚡

### Memory Usage:
- **Without Unsloth**: 15-16 GB (needs paid Colab)
- **With Unsloth**: 8-10 GB (works on free Colab!)
- **Savings**: ~40-50% less memory 💾

### Quality:
- Same or better than standard fine-tuning
- 85-95% valid JSON generation
- 70-80% correct node types

---

## 🔧 Optimization Tips

### For Faster Training:
```python
# Use packing for short sequences
packing = True  # Can be 5x faster!

# Increase batch size (if memory allows)
per_device_train_batch_size = 4  # instead of 2

# Reduce sequence length for simple workflows
max_seq_length = 1024  # instead of 2048
```

### For Better Quality:
```python
# More epochs
num_train_epochs = 5  # instead of 3

# Higher LoRA rank
r = 32  # instead of 16 (more parameters)

# Lower learning rate
learning_rate = 1e-4  # instead of 2e-4
```

### For Less Memory:
```python
# Gradient checkpointing
use_gradient_checkpointing = "unsloth"

# Smaller batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
```

---

## 🎓 Model Recommendations

### For Your n8n Project:

**1. Mistral 7B** (Recommended!)
```python
model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
```
- Best balance of speed and quality
- Great with structured JSON output
- Fits perfectly on free Colab

**2. Llama 3 8B** (Most Powerful)
```python
model_name = "unsloth/llama-3-8b-bnb-4bit"
```
- Best quality
- Slightly slower
- May need more memory

**3. Phi-3 Mini** (Fastest)
```python
model_name = "unsloth/phi-3-mini-4k-instruct-bnb-4bit"
```
- Super fast training
- Smaller model
- Good for prototyping

---

## 📝 Complete Workflow

### 1. Prepare Data (Local)
```bash
python prepare_training_data.py
```

### 2. Upload to Colab
- Go to https://colab.research.google.com
- Create new notebook
- Upload `training_data_alpaca.json`

### 3. Train with Unsloth (Colab)
- Copy the Colab notebook code above
- Run all cells
- Wait ~2-3 hours

### 4. Download Model (Colab)
```python
!zip -r n8n_generator.zip n8n_generator/
from google.colab import files
files.download('n8n_generator.zip')
```

### 5. Use Locally
```bash
# Extract model
unzip n8n_generator.zip

# Generate workflows
python generate_workflow.py \
    --model-type local \
    --model-path ./n8n_generator \
    --description "Your workflow description"
```

---

## 🚀 Advanced: Quantize for Production

After training, make it even faster:

```python
from unsloth import FastLanguageModel

# Load your model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "n8n_generator",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Save with different quantizations
model.save_pretrained_merged("n8n_generator_16bit", tokenizer, save_method="merged_16bit")
model.save_pretrained_merged("n8n_generator_4bit", tokenizer, save_method="merged_4bit")

# Or save to GGUF for llama.cpp
model.save_pretrained_gguf("n8n_generator_gguf", tokenizer, quantization_method="q4_k_m")
```

---

## 💡 Pro Tips

1. **Start Small**: Test with 500 examples first (5 min training)
2. **Monitor Loss**: Should decrease steadily
3. **Test Early**: Generate after 1 epoch to check quality
4. **Save Checkpoints**: In case training crashes
5. **Use Weights & Biases**: `report_to = "wandb"` for tracking

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

# Or reduce sequence length
max_seq_length = 1024
```

### Training Too Slow
```python
# Enable packing
packing = True

# Use smaller model
model_name = "unsloth/phi-3-mini-4k-instruct-bnb-4bit"
```

### Poor Quality
```python
# More training
num_train_epochs = 5

# Better learning rate
learning_rate = 1e-4

# Higher LoRA rank
r = 32
```

---

## 🎯 Why Unsloth is Perfect for Your Project

1. **Free Colab works** - No need for paid GPU
2. **Fast iteration** - Train, test, improve in hours not days
3. **Easy to use** - Less code than standard methods
4. **Great community** - Active support and examples
5. **Production ready** - Export to GGUF, 16bit, 4bit

---

## 📚 Resources

- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Unsloth Docs**: https://docs.unsloth.ai
- **Example Notebooks**: https://github.com/unslothai/unsloth/tree/main/notebooks
- **Discord**: https://discord.gg/unsloth

---

## 🎉 Quick Start Command

```bash
# 1. Prepare data
python prepare_training_data.py

# 2. Open Colab
# https://colab.research.google.com

# 3. Copy the Colab notebook code from above

# 4. Train (2-3 hours on free GPU)

# 5. Download and use!
```

**Training 6,000 n8n workflows in 2-3 hours on free Colab? That's Unsloth magic! ⚡**
