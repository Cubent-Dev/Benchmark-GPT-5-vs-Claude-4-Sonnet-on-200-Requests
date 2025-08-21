#!/usr/bin/env python3
"""
Script to run both GPT-5 and Claude 4 Sonnet on all evaluation prompts.
Measures latency and saves outputs for comparison.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List
import asyncio
from dataclasses import dataclass
from datetime import datetime

import openai
import anthropic
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelResponse:
    """Container for model response with metadata."""
    content: str
    latency: float
    timestamp: datetime
    model: str
    prompt_id: str
    error: str = None

class ModelRunner:
    """Handles running prompts against both models."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        self.prompts_dir = Path("prompts")
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        (self.outputs_dir / "gpt5").mkdir(exist_ok=True)
        (self.outputs_dir / "claude4").mkdir(exist_ok=True)
    
    def load_prompts(self) -> List[Dict[str, Any]]:
        """Load all prompt files."""
        prompts = []
        for prompt_file in sorted(self.prompts_dir.glob("*.json")):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                prompts.append(prompt_data)
        return prompts
    
    async def run_gpt5(self, prompt: str, prompt_id: str, domain: str) -> ModelResponse:
        """Run prompt against GPT-5."""
        try:
            # Adjust temperature based on domain
            temperature = 0.3 if domain in ["reasoning_math", "data_analysis", "knowledge_facts"] else 0.7
            
            start_time = time.time()
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-5",  # Assuming this is the model name
                messages=[
                    {"role": "system", "content": "You are a helpful, accurate, and thoughtful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1200
            )
            latency = time.time() - start_time
            
            return ModelResponse(
                content=response.choices[0].message.content,
                latency=latency,
                timestamp=datetime.now(),
                model="gpt-5",
                prompt_id=prompt_id
            )
        except Exception as e:
            return ModelResponse(
                content="",
                latency=0,
                timestamp=datetime.now(),
                model="gpt-5",
                prompt_id=prompt_id,
                error=str(e)
            )
    
    async def run_claude4(self, prompt: str, prompt_id: str, domain: str) -> ModelResponse:
        """Run prompt against Claude 4 Sonnet."""
        try:
            # Adjust temperature based on domain
            temperature = 0.3 if domain in ["reasoning_math", "data_analysis", "knowledge_facts"] else 0.7
            
            start_time = time.time()
            response = await self.anthropic_client.messages.acreate(
                model="claude-3-5-sonnet-20241022",  # Latest Claude model
                max_tokens=1200,
                temperature=temperature,
                system="You are a helpful, accurate, and thoughtful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            latency = time.time() - start_time
            
            return ModelResponse(
                content=response.content[0].text,
                latency=latency,
                timestamp=datetime.now(),
                model="claude-4-sonnet",
                prompt_id=prompt_id
            )
        except Exception as e:
            return ModelResponse(
                content="",
                latency=0,
                timestamp=datetime.now(),
                model="claude-4-sonnet",
                prompt_id=prompt_id,
                error=str(e)
            )
    
    def save_response(self, response: ModelResponse):
        """Save model response to file."""
        model_dir = "gpt5" if "gpt" in response.model.lower() else "claude4"
        output_file = self.outputs_dir / model_dir / f"{response.prompt_id}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {response.model}\n")
            f.write(f"Timestamp: {response.timestamp}\n")
            f.write(f"Latency: {response.latency:.2f}s\n")
            if response.error:
                f.write(f"Error: {response.error}\n")
            f.write(f"{'='*50}\n\n")
            f.write(response.content)
    
    async def run_evaluation(self):
        """Run full evaluation across all prompts."""
        prompts = self.load_prompts()
        print(f"Running evaluation on {len(prompts)} prompts...")
        
        results = []
        
        for prompt_data in tqdm(prompts, desc="Processing prompts"):
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["prompt"]
            domain = prompt_data["domain"]
            
            # Run both models concurrently
            gpt5_task = self.run_gpt5(prompt_text, prompt_id, domain)
            claude4_task = self.run_claude4(prompt_text, prompt_id, domain)
            
            gpt5_response, claude4_response = await asyncio.gather(gpt5_task, claude4_task)
            
            # Save responses
            self.save_response(gpt5_response)
            self.save_response(claude4_response)
            
            results.append({
                "prompt_id": prompt_id,
                "domain": domain,
                "gpt5_latency": gpt5_response.latency,
                "claude4_latency": claude4_response.latency,
                "gpt5_error": gpt5_response.error,
                "claude4_error": claude4_response.error
            })
            
            # Small delay to respect rate limits
            await asyncio.sleep(1)
        
        # Save latency results
        with open("results/latency_raw.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Evaluation complete!")

if __name__ == "__main__":
    runner = ModelRunner()
    asyncio.run(runner.run_evaluation())
