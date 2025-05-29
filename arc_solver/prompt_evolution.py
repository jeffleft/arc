from typing import List, Dict, Optional
import json
from openai import OpenAI

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
                    "content": "You are an expert at crafting prompts for solving ARC puzzles. Your task is to analyze the performance of previous prompts and suggest improvements."
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
        
        return response.choices[0].message.content
        
    def add_prompt(self, prompt: str, score: int, commentary: str):
        """Add a prompt and its performance to history."""
        self.prompt_history.append({
            "prompt": prompt,
            "score": score,
            "commentary": commentary
        })
        
    def _get_initial_prompt(self) -> str:
        """Generate the initial prompt."""
        return """### 1. **Cell-by-Cell Grounded Comparison**
- For **each training input/output pair**:
    - Inspect the grids **cell by cell**, not just by object or region.
    - For every grid location, record **what changes, what stays the same, and where each feature appears** - including color, label, connections, and background.
    - Note the **exact coordinates and arrangement** of every pattern, shape, line, path, or region.

### 2. **Explicit, Example-Based Rule Construction**
- **Before you use the tools, write a clear, stepwise rule** defining (in natural language):
    - **How to identify every relevant feature** (by color, shape, adjacency, position, connectivity, etc.).
    - **Exactly how each feature transforms or evolves**: Are elements preserved, moved, relabeled, copied, expanded,reflected, repeated, reordered, removed?
    - State explicitly **which positions are absolute (fixed coordinates)** and **which are relative (moved/shaped based on other features)**.
    - **Do not generalize, shift, align, or compress** unless every single training pair supports that operation.

### 3. **Cross-Example Rule Testing and Output Verification**
- **Apply your prospective rule** to every training pair; dry run in your head
    - **Double-check that every output cell exactly matches the example** in value, position, and shape.
    - If any difference (even a single cell) is found, **revise your rule to eliminate this discrepancy**.
    - Confirm that your logic never introduces, omits, shifts, or merges features in a way not shown in the outputs.

### 4. **Implementation - Faithful Output Construction**
- **Implement your stepwise rules exactly as written.**
- Produce output grids that **perfectly match the expected samples in content, size, structure, and position**.
- **Before finalizing**, always compare your output grid directly to the sample - cell by cell - to guarantee a perfect match.

**Self-Check Before Submission:**  
For each aspect of your solution, ask:  
- "Does this recreate exactly **what** changes and **where** - with no extra shifting, cropping, or alignment - just as shown in every training output, with no plausible alternatives?"  
If not, **return, reanalyze, and revise before proceeding.**
"""
        
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