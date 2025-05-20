import os
import json
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from solver import ARCSolver
from grid_ops import GridOperations
from prompt_evolution import PromptEvolution
import shutil
from datetime import datetime
import base64


def load_task(task_path: str) -> Dict:
    """Load a task JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def save_results(task_name: str, results_dir: str, task_results: Dict):
    """Save task results to the results directory."""
    # Create task directory
    task_dir = os.path.join(results_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Create a copy of task results without the solver instance
    serializable_results = task_results.copy()
    solver = serializable_results.pop("solver", None)
    
    # Save task results
    with open(os.path.join(task_dir, "results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save images
    for img_name, img_data in serializable_results["images"].items():
        img_path = os.path.join(task_dir, f"{img_name}.png")
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img_data))  # Convert base64 string to bytes
    
    # Save message history
    with open(os.path.join(task_dir, "message_history.json"), "w") as f:
        json.dump(serializable_results["message_history"], f, indent=2)
        
    # Save intermediate grid states
    intermediate_dir = os.path.join(task_dir, "intermediate_states")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Save intermediate states info
    with open(os.path.join(intermediate_dir, "states.json"), "w") as f:
        json.dump(serializable_results["intermediate_states"], f, indent=2)
    
    # Save intermediate grid images
    for state in serializable_results["intermediate_states"]:
        step = state["step"]
        grid = state["grid"]
        img_data = solver._grid_to_image(grid)  # Use solver instance for image generation
        
        # Save image
        img_path = os.path.join(intermediate_dir, f"step_{step:03d}.png")
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img_data))
        
        # Save grid data
        grid_path = os.path.join(intermediate_dir, f"step_{step:03d}.json")
        with open(grid_path, "w") as f:
            json.dump({
                "step": step,
                "grid": grid,
                "tool": state["tool"],
                "description": state["description"]
            }, f, indent=2)


def evaluate_solution(predicted: List[List[int]], expected: List[List[int]], confidence: int, plan: str, solver: ARCSolver) -> Tuple[int, str]:
    """Evaluate the solution accuracy and get LLM commentary.
    Returns (score, commentary) where score is 0 or 1."""
    if len(predicted) != len(expected) or len(predicted[0]) != len(expected[0]):
        return 0, "Grid dimensions do not match"
        
    # Check if all cells match
    is_correct = all(
        pred == exp
        for pred_row, exp_row in zip(predicted, expected)
        for pred, exp in zip(pred_row, exp_row)
    )
    
    # Get visual representations
    predicted_image = solver._grid_to_image(predicted)
    expected_image = solver._grid_to_image(expected)
    
    # Get LLM commentary
    response = solver.client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing ARC puzzle solutions. Your task is to explain why a solution is correct or incorrect, focusing on the underlying rules inferred."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Analyze this ARC puzzle solution:

Expected output (JSON):
{json.dumps(expected)}

Predicted output (JSON):
{json.dumps(predicted)}

(ground truth and predicted images are also attached.)

Model's confidence that it is correct: {confidence}/10

Model's initial plan for attacking the task:
{plan}

Please explain why the solution is {'correct' if is_correct else 'incorrect'}. Focus on:
1. The patterns and transformations that should have been applied
2. Where the solution {'matches' if is_correct else 'deviates from'} the expected pattern
3. Any insights about the underlying rule or concept
4. How the model went wrong/right with its plan"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{expected_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{predicted_image}"
                        }
                    }
                ]
            }
        ]
    )
    
    commentary = response.choices[0].message.content
    return 1 if is_correct else 0, commentary

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"training_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize components
    solver = ARCSolver(api_key, results_dir)
    prompt_evolution = PromptEvolution(api_key)
    
    # Save initial prompt
    with open(os.path.join(results_dir, "initial_prompt.txt"), "w") as f:
        f.write(prompt_evolution._get_initial_prompt())
    
    # Load tasks
    data_dir = "../data"
    training_dir = os.path.join(data_dir, "training")
    
    # Process each task (limited to 5)
    task_count = 0
    for task_file in os.listdir(training_dir):
        if not task_file.endswith(".json"):
            continue
            
        if task_count >= 5:
            break
            
        task_path = os.path.join(training_dir, task_file)
        task = load_task(task_path)
        
        # Get current prompt
        current_prompt = prompt_evolution.evolve_prompt()
        
        # Get the single test case
        test_case = task["test"][0]
        input_grid = test_case["input"]
        expected_output = test_case["output"]
        
        # Solve the task
        predicted_output, confidence = solver.solve_task(
            task, 
            input_grid, 
            current_prompt,
            task_file.replace(".json", "")
        )

        # Retrieve plan
        plan = [x for x in solver.get_message_history() if x.get("role") == "assistant"][0]["content"]
        
        # Evaluate the solution
        score, commentary = evaluate_solution(predicted_output, expected_output, confidence, plan, solver)
        
        # Update prompt evolution with score and commentary
        prompt_evolution.add_prompt(current_prompt, score, commentary)
        
        # Prepare task results
        task_results = {
            "task_name": task_file,
            "input_grid": input_grid,
            "expected_output": expected_output,
            "predicted_output": predicted_output,
            "score": score,
            "confidence": confidence,
            "commentary": commentary,
            "prompt": current_prompt,
            "images": {
                "input": solver._grid_to_image(input_grid),
                "expected": solver._grid_to_image(expected_output),
                "predicted": solver._grid_to_image(predicted_output)
            },
            "message_history": solver.get_message_history(),
            "intermediate_states": solver.get_intermediate_grids(),
            "solver": solver  # Pass solver instance for image generation
        }
        
        # Save results
        save_results(task_file.replace(".json", ""), results_dir, task_results)
        
        print(f"Task: {task_file}")
        print(f"Score: {score}")
        print(f"Confidence: {confidence:.2f}")
        print("\nAnalysis:")
        print(commentary)
        print("---")
        
        task_count += 1
    
    # Save final prompt evolution history
    with open(os.path.join(results_dir, "prompt_evolution_history.json"), "w") as f:
        json.dump(prompt_evolution.prompt_history, f, indent=2)

if __name__ == "__main__":
    main() 