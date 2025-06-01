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
        return """1. Map differences
   Compare each training output to its input cell-by-cell. Record every changed cell with coordinates, old and new values, neighbour context, and note unchanged areas. Map which visible regions alter, which remain, and where boundaries hold. Make no inferences—only observe.

2. Derive rules solely from observed changes
   Build rules that explain every changed cell and nothing else. Precisely state how each target is recognised (colour, local pattern, location, structure) and where a rule must stop; never act on regions not shown unequivocally in all examples. Before accepting a rule, ask if it ever edits a wrong cell or misses a required one; if so, narrow or split it. Boundaries such as connectors are hard walls whenever they block change even once.

3. Validate through full simulation
   Apply each rule to every training pair. Reject or tighten if it adds, omits, merges, leaks, or crosses a forbidden boundary even once. Prefer multiple narrow rules over one broad compressive rule.

4. Implement conservatively
   Code only fully validated rules; do not generalise, smooth, or fill unless every example demands it. When uncertain, edit less (underfit) rather than risk over-reach. Re-check that each rule respects its stopping conditions.

5. Final consistency check
   Compare generated outputs to provided ones cell-by-cell and edge-to-edge. Any mismatch, however small, sends you back to refine rules; never patch outputs. Justify every alteration with explicit evidence common to all examples.

Core principles
Let outputs, not intuition, dictate rules. Treat any connector or thin boundary that blocks change in one example as an absolute barrier. Every rule must answer: “Does this cover all and only the required changes across every training case?”

Goal
Reproduce every training output exactly—contents, boundaries, and preserved features—via difference-driven mapping, minimal exceptionless rules, strict boundary adherence, and exhaustive validation.
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