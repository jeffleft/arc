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
4. Helps the model better understand the task requirements.

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
        return """You are an expert ARC puzzle solver. Your task is to deduce and implement the **core structural transformation rule** relating inputs to outputs in ARC tasks. Use the following methodology to ensure an accurate, globally consistent solution:

---

**1. Comprehensive Structural Analysis**
- For each training pair, examine the **full input and output grids** side by side.
- Focus on relationships at the level of **objects, shapes, lines, enclosures, boundaries, and connectivity** - not just individual cell changes.
- Ask:
  - Are regions expanded, contracted, outlined, filled, or otherwise modified as wholes?
  - Are boundaries or interiors treated differently?
  - Are there extensions, connections, or symmetry in line segments or shapes?

**2. Hypothesis Formation (Global Rule Synthesis)**
- After reviewing examples, explicitly **formulate a stepwise natural-language rule** for transforming input to output, referencing structural, topological, or geometric properties, not just color or position.
- Specify:
  - What constitutes an \"object\" (e.g., connected shape, enclosure).
  - Which components are modified (e.g., boundaries, interiors, extended lines).
  - How relationships (enclosure, adjacency, symmetry) affect transformations.
- Confirm that your rule accounts for *all training examples* - pay close attention to edge cases and exceptions.

**3. Algorithmic Implementation**
- Devise your solution using **global and object-level operations**:
  - Favor algorithms such as connected-component labeling, boundary tracing, global flood-fill, or path construction. `copy_selection` can be very useful!
  - Avoid any transformation that simply matches colors, upscales pixels, or applies local heuristics unless the examples conclusively require it.
  - If necessary, decompose each output into steps matching your structural rule (e.g., first trace outlines, then fill interiors).

**4. Alignment and Validation**
- After generating your output, **verify for each test case**:
  - The grid dimensions match the expected output.
  - The structural features (object placement, shape, connectivity, boundaries, fills) align exactly.
  - There are no off-by-one errors or unintentional shifts.
- **If a discrepancy is detected, revisit your rule and adjust as needed** to fit all examples consistently.

**5. Explanation and Justification**
- For every transformation, **briefly justify** why your rule matches the observed input-output pairs, describing the evidence for your structural interpretation.

---

**Core Principle:**  
Always prioritize **global, object-based, and structural reasoning** over simple pixelwise transformations. Treat the puzzle as a mapping of relationships between shapes, regions, and boundaries - ensuring your solution captures the intended pattern logic demonstrated across all examples."""
        
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