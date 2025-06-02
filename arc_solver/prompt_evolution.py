from typing import List, Dict, Optional
import json
from openai import OpenAI
import os
from datetime import datetime

class PromptEvolution:
    def __init__(self, api_key: str):
        """Initialize the prompt evolution system."""
        self.client = OpenAI(api_key=api_key)
        self.prompt_history = []
        
    def evolve_prompt(self) -> str:
        """Generate a new prompt based on history."""
        if not self.prompt_history:
            return self._get_initial_prompt()
            
        # Analyze history to improve prompt
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at crafting prompts for solving ARC puzzles. Your task is to analyze the performance of previous prompts and output a new prompt that improves on the previous ones."
                },
                {
                    "role": "user",
                    "content": f"""Analyze these previous prompts and their performance:

{self._format_history()}

Based on this analysis, suggest a new prompt that:
1. Addresses the weaknesses identified in previous attempts
2. Builds on successful strategies
3. Provides clearer guidance for pattern recognition
4. Helps the model better understand the task requirements

Output just the new prompt!"""
                }
            ]
        )
        
        new_prompt = response.choices[0].message.content

        # Save the evolved prompt to a timestamped file
        try:
            evolved_prompts_dir = os.path.join(os.path.dirname(__file__), "prompts", "evolved")
            os.makedirs(evolved_prompts_dir, exist_ok=True) # exist_ok=True ensures it doesn't error if dir exists

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_{timestamp}.txt"
            filepath = os.path.join(evolved_prompts_dir, filename)

            with open(filepath, "w") as f:
                f.write(new_prompt)
            # print(f"Saved evolved prompt to {filepath}") # Optional: for debugging
        except Exception as e:
            print(f"Error saving evolved prompt: {e}") # Log error but don't crash

        return new_prompt
        
    def add_prompt(self, prompt: str, score: int, commentary: str):
        """Add a prompt and its performance to history."""
        self.prompt_history.append({
            "prompt": prompt,
            "score": score,
            "commentary": commentary
        })
        
    def _get_initial_prompt(self) -> str:
        """Load the initial prompt from a file."""
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "initial_prompt.txt")
        try:
            with open(prompt_file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Initial prompt file not found at {prompt_file_path}")
            # Return a fallback prompt or raise an error
            return "Fallback initial prompt: Describe the transformations needed to get from input to output."
        except Exception as e:
            print(f"Error reading initial prompt file {prompt_file_path}: {e}")
            return "Fallback initial prompt due to error: Describe transformations."
        
    def _format_history(self) -> str:
        """Format prompt history for analysis."""
        history = []
        for i, entry in enumerate(self.prompt_history, 1):
            history.append(f"Attempt {i}:")
            history.append(f"Prompt: {entry['prompt']}")
            history.append(f"Score: {entry['score']}")
            history.append(f"Analysis: {entry['commentary']}")
            history.append("")
        return "\n".join(history) 