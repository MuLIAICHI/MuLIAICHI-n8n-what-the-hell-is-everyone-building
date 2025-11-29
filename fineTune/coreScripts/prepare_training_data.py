#!/usr/bin/env python3
"""
n8n Workflow to LLM Training Data Converter
Prepares workflow data for fine-tuning an LLM to generate workflows from natural language
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


class WorkflowToTrainingData:
    def __init__(self, workflows_file: str):
        """Initialize with scraped workflows data"""
        self.workflows_file = Path(workflows_file)
        
        with open(self.workflows_file, 'r', encoding='utf-8') as f:
            self.workflows = json.load(f)
        
        print(f"Loaded {len(self.workflows)} workflows")
    
    def clean_description(self, text: str) -> str:
        """Clean and normalize description text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (we don't need them for training)
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def extract_workflow_structure(self, workflow: Dict) -> Dict:
        """Extract essential workflow structure without sensitive data"""
        nodes = workflow.get('nodes', [])
        
        simplified_nodes = []
        for node in nodes:
            simplified_node = {
                'name': node.get('name', ''),
                'type': node.get('displayName', ''),
                'category': [cat.get('name', '') for cat in node.get('nodeCategories', [])]
            }
            simplified_nodes.append(simplified_node)
        
        return {
            'nodes': simplified_nodes,
            'node_count': len(simplified_nodes),
            'node_types': [n['type'] for n in simplified_nodes]
        }
    
    def create_training_pairs(self, min_description_length: int = 50) -> List[Dict]:
        """Create instruction-output pairs for training"""
        training_data = []
        
        for workflow in self.workflows:
            name = workflow.get('name', '')
            description = self.clean_description(workflow.get('description', ''))
            
            # Skip workflows without good descriptions
            if len(description) < min_description_length:
                continue
            
            # Extract workflow structure
            workflow_structure = self.extract_workflow_structure(workflow)
            
            # Create instruction prompt
            instruction = f"Create an n8n workflow for: {name}"
            if description:
                instruction += f"\n\nDescription: {description}"
            
            # Create the training pair
            training_pair = {
                'instruction': instruction,
                'input': '',
                'output': json.dumps(workflow_structure, indent=2)
            }
            
            training_data.append(training_pair)
        
        return training_data
    
    def create_alpaca_format(self, output_file: str = 'training_data_alpaca.json'):
        """Create training data in Alpaca format (popular for fine-tuning)"""
        training_pairs = self.create_training_pairs()
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Created Alpaca format training data: {output_path}")
        print(f"✓ Total training examples: {len(training_pairs)}")
        
        return output_path
    
    def create_openai_format(self, output_file: str = 'training_data_openai.jsonl'):
        """Create training data in OpenAI fine-tuning format"""
        training_pairs = self.create_training_pairs()
        
        openai_format = []
        for pair in training_pairs:
            openai_format.append({
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an n8n workflow expert. Generate workflow structures based on user requirements.'
                    },
                    {
                        'role': 'user',
                        'content': pair['instruction']
                    },
                    {
                        'role': 'assistant',
                        'content': pair['output']
                    }
                ]
            })
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in openai_format:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Created OpenAI format training data: {output_path}")
        print(f"✓ Total training examples: {len(openai_format)}")
        
        return output_path
    
    def create_simple_pairs_format(self, output_file: str = 'training_data_simple.json'):
        """Create simple prompt-completion pairs"""
        training_pairs = self.create_training_pairs()
        
        simple_format = []
        for pair in training_pairs:
            simple_format.append({
                'prompt': pair['instruction'],
                'completion': pair['output']
            })
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simple_format, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Created simple format training data: {output_path}")
        print(f"✓ Total training examples: {len(simple_format)}")
        
        return output_path
    
    def analyze_training_data(self):
        """Analyze the quality and characteristics of training data"""
        training_pairs = self.create_training_pairs()
        
        print("\n" + "=" * 60)
        print("TRAINING DATA ANALYSIS")
        print("=" * 60)
        
        # Description lengths
        desc_lengths = [len(pair['instruction']) for pair in training_pairs]
        avg_desc_length = sum(desc_lengths) / len(desc_lengths)
        
        print(f"\nTotal training examples: {len(training_pairs)}")
        print(f"Average instruction length: {avg_desc_length:.0f} characters")
        print(f"Min instruction length: {min(desc_lengths)}")
        print(f"Max instruction length: {max(desc_lengths)}")
        
        # Node counts
        node_counts = []
        for pair in training_pairs:
            workflow = json.loads(pair['output'])
            node_counts.append(workflow['node_count'])
        
        print(f"\nAverage nodes per workflow: {sum(node_counts) / len(node_counts):.1f}")
        print(f"Min nodes: {min(node_counts)}")
        print(f"Max nodes: {max(node_counts)}")
        
        # Most common node types
        from collections import Counter
        all_nodes = []
        for pair in training_pairs:
            workflow = json.loads(pair['output'])
            all_nodes.extend(workflow['node_types'])
        
        node_counts_dict = Counter(all_nodes)
        
        print(f"\nTop 10 most common nodes in training data:")
        for node, count in node_counts_dict.most_common(10):
            print(f"  {node}: {count}")
    
    def create_all_formats(self):
        """Generate all training data formats"""
        print("\n" + "=" * 60)
        print("GENERATING TRAINING DATA IN MULTIPLE FORMATS")
        print("=" * 60)
        
        # Create output directory
        output_dir = Path('n8n_data/training_data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all formats
        self.create_alpaca_format(output_dir / 'training_data_alpaca.json')
        self.create_openai_format(output_dir / 'training_data_openai.jsonl')
        self.create_simple_pairs_format(output_dir / 'training_data_simple.json')
        
        # Analyze
        self.analyze_training_data()
        
        print("\n" + "=" * 60)
        print("✓ ALL TRAINING DATA GENERATED")
        print("=" * 60)
        print(f"\nFiles saved to: {output_dir}")


def main():
    """Main execution"""
    # Find the most recent workflow data
    data_dir = Path('n8n_data/raw')
    print(data_dir.absolute())
    
    if not data_dir.exists():
        print("Error: No workflow data found. Run the scraper first!")
        return
    
    workflow_files = list(data_dir.glob('workflows_*.json'))
    
    if not workflow_files:
        print("Error: No workflow data files found. Run the scraper first!")
        return
    
    # Use most recent file
    latest_file = max(workflow_files, key=lambda p: p.stat().st_mtime)
    print(f"Using workflow data: {latest_file}")
    
    # Create training data
    converter = WorkflowToTrainingData(latest_file)
    converter.create_all_formats()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("\n1. Choose your fine-tuning approach:")
    print("   - OpenAI fine-tuning (easiest, costs money)")
    print("   - Open source models (free, more work)")
    print("   - Local fine-tuning with LoRA/QLoRA")
    print("\n2. Use the appropriate training data format:")
    print("   - training_data_openai.jsonl → OpenAI API")
    print("   - training_data_alpaca.json → Llama, Mistral, etc.")
    print("   - training_data_simple.json → Custom training")
    print("\n3. See fine_tune_guide.md for detailed instructions")


if __name__ == '__main__':
    main()
