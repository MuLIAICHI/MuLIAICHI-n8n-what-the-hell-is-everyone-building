import sys
import types

# Mock audioop module for Python 3.13 compatibility
if sys.version_info >= (3, 13):
    try:
        import audioop
    except ImportError:
        # Create a dummy module
        mock_audioop = types.ModuleType("audioop")
        
        # Add dummy functions that might be called (add more if needed)
        def dummy_func(*args, **kwargs):
            return b""
            
        mock_audioop.lin2adpcm = dummy_func
        mock_audioop.adpcm2lin = dummy_func
        mock_audioop.lin2ulaw = dummy_func
        mock_audioop.ulaw2lin = dummy_func
        mock_audioop.lin2alaw = dummy_func
        mock_audioop.alaw2lin = dummy_func
        mock_audioop.rms = lambda *args: 0
        mock_audioop.avg = lambda *args: 0
        mock_audioop.max = lambda *args: 0
        mock_audioop.minmax = lambda *args: (0, 0)
        mock_audioop.avgpp = lambda *args: 0
        mock_audioop.maxpp = lambda *args: 0
        mock_audioop.cross = lambda *args: 0
        mock_audioop.mul = dummy_func
        mock_audioop.tomono = dummy_func
        mock_audioop.tostereo = dummy_func
        mock_audioop.add = dummy_func
        mock_audioop.bias = dummy_func
        mock_audioop.reverse = dummy_func
        mock_audioop.byteswap = dummy_func
        mock_audioop.ratecv = lambda *args: (b"", 0)
        
        # Inject into sys.modules
        sys.modules["audioop"] = mock_audioop
        sys.modules["pyaudioop"] = mock_audioop  # Also mock pyaudioop just in case

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the model and tokenizer from HuggingFace"""
    global model, tokenizer
    
    print("🔄 Loading model from HuggingFace...")
    
    model_name = "MustaphaL/n8n-workflow-generator"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print("✅ Model loaded successfully!")
    return "Model loaded and ready!"

def generate_workflow(
    user_prompt,
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    progress=gr.Progress()
):
    """Generate n8n workflow from user prompt"""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Error: Model not loaded! Click 'Load Model' first.", "", ""
    
    try:
        progress(0.1, desc="Preparing prompt...")
        
        # Alpaca format
        alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
        
        # Format the instruction
        instruction = f"Create an n8n workflow for: {user_prompt}"
        
        # Tokenize
        progress(0.3, desc="Tokenizing...")
        inputs = tokenizer(
            [alpaca_prompt.format(instruction, "")],
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        progress(0.5, desc="Generating workflow...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        progress(0.8, desc="Processing output...")
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        response = result.split("### Response:")[-1].strip()
        
        progress(1.0, desc="Complete!")
        
        # Try to parse as JSON
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                workflow_json = json_match.group()
                parsed = json.loads(workflow_json)
                formatted_json = json.dumps(parsed, indent=2)
                status = "✅ Valid n8n workflow generated!"
                return status, response, formatted_json
            else:
                status = "⚠️ Generated but no valid JSON found"
                return status, response, "No JSON structure detected"
        except json.JSONDecodeError:
            status = "⚠️ Generated but JSON parsing failed"
            return status, response, "Could not parse as valid JSON"
            
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# Example prompts
examples = [
    ["Build a Telegram chatbot that uses OpenAI to respond to messages"],
    ["Create a workflow that monitors a Gmail inbox and sends Slack notifications for important emails"],
    ["Build an automation that scrapes product prices from a website and saves them to Google Sheets"],
    ["Create a workflow that posts Twitter updates to LinkedIn automatically"],
    ["Build a system that processes invoices from email attachments and updates Airtable"],
    ["Create a workflow that monitors GitHub issues and creates Trello cards"],
    ["Build an automation that sends daily weather reports via email"],
    ["Create a workflow that backs up Discord messages to a database"],
]

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.info-box {
    background: #f0f7ff;
    border-left: 4px solid #667eea;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
"""

# Build the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="n8n Workflow Generator") as app:
    
    # Header
    gr.HTML("""
    <div class="header">
        <h1>🤖 n8n Workflow Generator</h1>
        <p>AI-powered workflow generation using Llama 3 8B fine-tuned on 6,000+ n8n workflows</p>
    </div>
    """)
    
    # Info section
    gr.Markdown("""
    <div class="info-box">
    
    ## 📚 About This Model
    
    This model was fine-tuned on **6,000+ real n8n workflows** scraped from the marketplace.
    
    **Read the full story:**
    - 📝 [Part 1: What Are People Building?](https://medium.com/@mustaphaliaichi/what-are-people-actually-building-in-n8n-i-scraped-over-6-000-workflows-to-find-out-59eb8e34c317)
    - 📝 [Part 2: Fine-tuning Llama 3 After Mistral Failed](https://medium.com/@mustaphaliaichi/i-fine-tuned-llama-3-on-6-000-n8n-workflows-after-mistral-failed-spectacularly-927cce57df14)
    - 🤗 [Model on HuggingFace](https://huggingface.co/MustaphaL/n8n-workflow-generator)
    
    **Created by:** [Mustapha LIAICHI](https://n8nlearninghub.com) | [Reddit: r/n8nLearningHub](https://reddit.com/r/n8nLearningHub)
    
    </div>
    """)
    
    # Load model button
    with gr.Row():
        load_btn = gr.Button("🚀 Load Model (Click First!)", variant="primary", size="lg")
        load_status = gr.Textbox(label="Status", interactive=False, value="Model not loaded yet...")
    
    load_btn.click(fn=load_model, outputs=load_status)
    
    gr.Markdown("---")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 💬 Describe Your Workflow")
            
            user_input = gr.Textbox(
                label="What do you want to automate?",
                placeholder="Example: Build a Telegram chatbot that uses OpenAI to respond to messages",
                lines=3
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Max Tokens",
                    info="Maximum length of generated workflow"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling parameter"
                )
            
            generate_btn = gr.Button("✨ Generate Workflow", variant="primary", size="lg")
            
            gr.Markdown("### 💡 Try These Examples")
            gr.Examples(
                examples=examples,
                inputs=user_input,
                label=None
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Generated Workflow")
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1
            )
            
            with gr.Tabs():
                with gr.Tab("📝 Raw Output"):
                    raw_output = gr.Textbox(
                        label="Generated Text",
                        lines=15,
                        max_lines=20,
                        show_copy_button=True
                    )
                
                with gr.Tab("🔍 JSON View"):
                    json_output = gr.Code(
                        label="Formatted JSON",
                        language="json",
                        lines=15
                    )
    
    # Connect generate button
    generate_btn.click(
        fn=generate_workflow,
        inputs=[user_input, max_tokens, temperature, top_p],
        outputs=[status_output, raw_output, json_output]
    )
    
    # Footer
    gr.Markdown("""
    ---
    
    ### 🔗 Resources
    
    - 📚 [n8n Learning Hub](https://n8nlearninghub.com) - Tutorials and guides
    - 💬 [Reddit Community](https://reddit.com/r/n8nLearningHub) - Join 1,000+ members
    - 🤗 [Dataset](https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data) - Training data
    - 📓 [Colab Notebook](https://colab.research.google.com) - Train your own
    
    ### ⚠️ Disclaimer
    
    This is an AI-generated workflow generator. Always review and test generated workflows before deploying to production.
    The model may occasionally generate workflows that need manual adjustment.
    
    ### 📝 How to Use Generated Workflows
    
    1. Copy the JSON output
    2. Open n8n editor
    3. Create new workflow
    4. Import JSON
    5. Configure credentials
    6. Test and deploy!
    
    ---
    
    **Built with ❤️ by MHL** | **Powered by Llama 3 8B + Unsloth**
    """)

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re

# Global variables
model = None
tokenizer = None

def load_model_optimized():
    """Load model with 4-bit quantization - MUCH smaller memory footprint"""
    global model, tokenizer
    
    try:
        print("🔄 Starting model load...")
        
        model_name = "MustaphaL/n8n-workflow-generator"
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("📥 Loading model (4-bit quantized)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
        return "✅ Model loaded! Ready to generate workflows."
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return error_msg

def generate_workflow(
    user_prompt,
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    progress=gr.Progress()
):
    """Generate n8n workflow from user prompt"""
    
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Error: Model not loaded! Click 'Load Model' first.", "", ""
    
    try:
        progress(0.1, desc="Preparing prompt...")
        
        # Alpaca format
        alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
        
        instruction = f"Create an n8n workflow for: {user_prompt}"
        
        progress(0.3, desc="Tokenizing...")
        inputs = tokenizer(
            [alpaca_prompt.format(instruction, "")],
            return_tensors="pt"
        ).to(model.device)
        
        progress(0.5, desc="Generating workflow...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        progress(0.8, desc="Processing output...")
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = result.split("### Response:")[-1].strip()
        
        progress(1.0, desc="Complete!")
        
        # Try to parse JSON
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                workflow_json = json_match.group()
                parsed = json.loads(workflow_json)
                formatted_json = json.dumps(parsed, indent=2)
                status = "✅ Valid n8n workflow generated!"
                return status, response, formatted_json
            else:
                status = "⚠️ Generated but no valid JSON found"
                return status, response, "No JSON structure detected"
        except json.JSONDecodeError:
            status = "⚠️ Generated but JSON parsing failed"
            return status, response, "Could not parse as valid JSON"
            
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# Example prompts
examples = [
    ["Build a Telegram chatbot that uses OpenAI to respond to messages"],
    ["Create a workflow that monitors Gmail and sends Slack notifications"],
    ["Build an automation that scrapes prices and saves to Google Sheets"],
]

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
"""

# Build the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="n8n Workflow Generator") as app:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>🤖 n8n Workflow Generator</h1>
        <p>AI-powered workflow generation using Llama 3 8B (4-bit quantized for efficiency)</p>
    </div>
    """)
    
    gr.Markdown("""
    ## ⚡ Optimized for Low-Memory Systems
    
    This version uses 4-bit quantization to run on systems with limited RAM/VRAM.
    """)
    
    # Load model section
    with gr.Row():
        load_btn = gr.Button("🚀 Load Model (Click First!)", variant="primary", size="lg")
        load_status = gr.Textbox(label="Status", interactive=False, value="Click 'Load Model' to start...")
    
    load_btn.click(fn=load_model_optimized, outputs=load_status)
    
    gr.Markdown("---")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 💬 Describe Your Workflow")
            
            user_input = gr.Textbox(
                label="What do you want to automate?",
                placeholder="Example: Build a Telegram chatbot that uses OpenAI",
                lines=3
            )
            
            with gr.Accordion("⚙️ Settings", open=False):
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=512,
                    value=256,
                    step=64,
                    label="Max Tokens (lower = faster)"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P"
                )
            
            generate_btn = gr.Button("✨ Generate Workflow", variant="primary", size="lg")
            
            gr.Examples(
                examples=examples,
                inputs=user_input,
                label="💡 Try These"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Generated Workflow")
            
            status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tabs():
                with gr.Tab("📝 Raw Output"):
                    raw_output = gr.Textbox(label="Generated Text", lines=12, show_copy_button=True)
                
                with gr.Tab("🔍 JSON View"):
                    json_output = gr.Code(label="Formatted JSON", language="json", lines=12)
    
    generate_btn.click(
        fn=generate_workflow,
        inputs=[user_input, max_tokens, temperature, top_p],
        outputs=[status_output, raw_output, json_output]
    )
    
    gr.Markdown("""
    ---
    ### 🔗 Resources
    - [Model](https://huggingface.co/MustaphaL/n8n-workflow-generator)
    - [n8n Learning Hub](https://n8nlearninghub.com)
    - [Reddit Community](https://reddit.com/r/n8nLearningHub)
    """)

# Launch configuration
if __name__ == "__main__":
    app.launch(
        share=False,
        server_name="localhost",
        server_port=7860,
        show_error=True
    )
