#!/usr/bin/env python3
"""
n8n Workflow Generator - Inference Script
Use your fine-tuned model to generate workflows from natural language
"""

import json
import argparse
from pathlib import Path


class N8nWorkflowGenerator:
    def __init__(self, model_type: str = "openai", model_path: str = None):
        """
        Initialize the workflow generator
        
        Args:
            model_type: 'openai', 'huggingface', 'local', or 'unsloth'
            model_path: Path to fine-tuned model (for local/huggingface/unsloth)
        """
        self.model_type = model_type
        self.model_path = model_path
        
        if model_type == "openai":
            self._init_openai()
        elif model_type == "huggingface":
            self._init_huggingface()
        elif model_type == "local":
            self._init_local()
        elif model_type == "unsloth":
            self._init_unsloth()
    
    def _init_openai(self):
        """Initialize OpenAI model"""
        try:
            import openai
            self.client = openai
            print("✓ OpenAI client initialized")
            print("Make sure to set OPENAI_API_KEY environment variable")
        except ImportError:
            print("Error: Install openai package: pip install openai")
    
    def _init_huggingface(self):
        """Initialize Hugging Face model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"Loading model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✓ Hugging Face model loaded")
        except ImportError:
            print("Error: Install transformers: pip install transformers torch")
    
    def _init_local(self):
        """Initialize local model (LoRA adapters)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            
            print(f"Loading LoRA adapters from: {self.model_path}")
            base_model = "mistralai/Mistral-7B-v0.1"  # Adjust as needed
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            print("✓ Local model with LoRA adapters loaded")
        except ImportError:
            print("Error: Install peft: pip install peft transformers torch")
    
    def _init_unsloth(self):
        """Initialize Unsloth fine-tuned model"""
        try:
            from unsloth import FastLanguageModel
            
            print(f"Loading Unsloth model from: {self.model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = self.model_path,
                max_seq_length = 2048,
                dtype = None,
                load_in_4bit = True,
            )
            
            # Enable fast inference
            FastLanguageModel.for_inference(self.model)
            print("✓ Unsloth model loaded with fast inference enabled")
        except ImportError:
            print("Error: Install unsloth: pip install unsloth")
        except Exception as e:
            print(f"Error loading Unsloth model: {e}")
    
    def generate_workflow_openai(self, description: str, model_id: str) -> dict:
        """Generate workflow using OpenAI fine-tuned model"""
        response = self.client.ChatCompletion.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an n8n workflow expert. Generate workflow structures based on user requirements."
                },
                {
                    "role": "user",
                    "content": f"Create an n8n workflow for: {description}"
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        output = response.choices[0].message.content
        return self._parse_workflow_output(output)
    
    def generate_workflow_huggingface(self, description: str) -> dict:
        """Generate workflow using Hugging Face model"""
        prompt = f"""### Instruction:
Create an n8n workflow for: {description}

### Response:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in output:
            output = output.split("### Response:")[1].strip()
        
        return self._parse_workflow_output(output)
    
    def _parse_workflow_output(self, output: str) -> dict:
        """Parse model output into workflow structure"""
        try:
            # Try to find JSON in the output
            start = output.find('{')
            end = output.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = output[start:end]
                workflow = json.loads(json_str)
                return {
                    'success': True,
                    'workflow': workflow,
                    'raw_output': output
                }
            else:
                return {
                    'success': False,
                    'error': 'No JSON found in output',
                    'raw_output': output
                }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Invalid JSON: {str(e)}',
                'raw_output': output
            }
    
    def generate(self, description: str, **kwargs) -> dict:
        """Generate workflow based on description"""
        print(f"\n{'='*60}")
        print(f"Generating workflow for: {description}")
        print(f"{'='*60}\n")
        
        if self.model_type == "openai":
            model_id = kwargs.get('model_id', 'ft:gpt-3.5-turbo:your-model-id')
            result = self.generate_workflow_openai(description, model_id)
        elif self.model_type in ["huggingface", "local", "unsloth"]:
            result = self.generate_workflow_huggingface(description)
        else:
            return {'success': False, 'error': 'Invalid model type'}
        
        if result['success']:
            print("✓ Workflow generated successfully!\n")
            print(json.dumps(result['workflow'], indent=2))
        else:
            print(f"✗ Generation failed: {result['error']}\n")
            print("Raw output:")
            print(result['raw_output'])
        
        return result
    
    def batch_generate(self, descriptions: list, output_file: str = None):
        """Generate multiple workflows"""
        results = []
        
        for i, desc in enumerate(descriptions, 1):
            print(f"\n[{i}/{len(descriptions)}] Processing: {desc[:50]}...")
            result = self.generate(desc)
            results.append({
                'description': desc,
                'success': result['success'],
                'workflow': result.get('workflow'),
                'error': result.get('error')
            })
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"BATCH GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(results)}")
        print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
        print(f"Failed: {len(results) - successful}")
        
        return results


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Generate n8n workflows from natural language')
    parser.add_argument('--model-type', choices=['openai', 'huggingface', 'local', 'unsloth'], 
                        default='openai', help='Type of model to use')
    parser.add_argument('--model-path', type=str, help='Path to model (for huggingface/local/unsloth)')
    parser.add_argument('--model-id', type=str, help='OpenAI model ID (for openai)')
    parser.add_argument('--description', type=str, help='Workflow description')
    parser.add_argument('--batch', type=str, help='File with descriptions (one per line)')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = N8nWorkflowGenerator(
        model_type=args.model_type,
        model_path=args.model_path
    )
    
    # Single generation
    if args.description:
        kwargs = {}
        if args.model_id:
            kwargs['model_id'] = args.model_id
        generator.generate(args.description, **kwargs)
    
    # Batch generation
    elif args.batch:
        with open(args.batch, 'r') as f:
            descriptions = [line.strip() for line in f if line.strip()]
        generator.batch_generate(descriptions, args.output)
    
    # Interactive mode
    else:
        print("\n" + "="*60)
        print("n8n WORKFLOW GENERATOR - Interactive Mode")
        print("="*60)
        print("\nEnter workflow descriptions (or 'quit' to exit)")
        
        while True:
            try:
                desc = input("\n> Describe your workflow: ").strip()
                
                if desc.lower() in ['quit', 'exit', 'q']:
                    break
                
                if desc:
                    kwargs = {}
                    if args.model_id:
                        kwargs['model_id'] = args.model_id
                    generator.generate(desc, **kwargs)
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break


if __name__ == '__main__':
    main()
