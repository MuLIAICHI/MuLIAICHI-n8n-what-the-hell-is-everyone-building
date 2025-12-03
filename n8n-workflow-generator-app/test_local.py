#!/usr/bin/env python3
"""
Quick Test Script for n8n Workflow Generator
Run this locally to test the model before deploying
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def test_model():
    """Test the model with a simple prompt"""
    
    print("🔄 Loading model...")
    model_name = "MustaphaL/n8n-workflow-generator"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    print("✅ Model loaded!")
    print(f"Device: {model.device}")
    
    # Alpaca format
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
    
    # Test prompt
    instruction = "Create an n8n workflow for: Build a Telegram chatbot that uses OpenAI to respond to messages"
    
    print(f"\n📝 Testing with prompt: {instruction}\n")
    
    # Tokenize
    inputs = tokenizer(
        [alpaca_prompt.format(instruction, "")],
        return_tensors="pt"
    ).to(model.device)
    
    print("🤖 Generating workflow...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = result.split("### Response:")[-1].strip()
    
    print("\n" + "="*80)
    print("GENERATED WORKFLOW:")
    print("="*80)
    print(response)
    print("="*80)
    
    # Try to parse JSON
    try:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            workflow_json = json_match.group()
            parsed = json.loads(workflow_json)
            print("\n✅ Valid JSON detected!")
            print("\nFormatted JSON:")
            print(json.dumps(parsed, indent=2))
        else:
            print("\n⚠️ No JSON structure found in output")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parsing failed: {e}")
    
    return response

if __name__ == "__main__":
    print("=" * 80)
    print("n8n WORKFLOW GENERATOR - LOCAL TEST")
    print("=" * 80)
    print()
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU (will be slower)")
    
    print()
    
    try:
        result = test_model()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
