import os
import json
from typing import Dict, List
from dotenv import load_dotenv
from multi_agent_solver import MultiAgentARCSolver
from datetime import datetime

def load_task(task_path: str) -> Dict:
    """Load a task JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)

def test_multi_agent_solver():
    """Test the multi-agent solver on a sample task."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"multi_agent_test_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize multi-agent solver
    solver = MultiAgentARCSolver(api_key, results_dir)
    
    # Load a sample task (adjust path as needed)
    data_dir = "../data/v2"
    training_dir = os.path.join(data_dir, "training")
    
    # Find the first available task
    task_file = None
    for f in os.listdir(training_dir):
        if f.endswith(".json"):
            task_file = f
            break
    
    if not task_file:
        print("No task files found in the training directory")
        return
    
    task_path = os.path.join(training_dir, task_file)
    task = load_task(task_path)
    
    print(f"Testing multi-agent solver on task: {task_file}")
    print(f"Number of training examples: {len(task.get('train', []))}")
    
    # Get the test case
    test_case = task["test"][0]
    input_grid = test_case["input"]
    expected_output = test_case["output"]
    
    print(f"Input grid size: {len(input_grid)}x{len(input_grid[0])}")
    print(f"Expected output size: {len(expected_output)}x{len(expected_output[0])}")
    
    # Define some example instructions
    evolved_instructions = """
    When solving ARC puzzles, follow these guidelines:
    1. Carefully analyze all training examples to identify patterns
    2. Look for transformations involving colors, shapes, positions, and relationships
    3. Consider symmetries, rotations, reflections, and scaling
    4. Pay attention to object boundaries, connectivity, and spatial relationships
    5. Use the most appropriate tools for the identified transformation
    6. Test your solution approach on training examples before applying to the test case
    """
    
    # Solve the task using multi-agent approach
    try:
        predicted_output, confidence = solver.solve_task(
            task, 
            input_grid, 
            evolved_instructions,
            task_file.replace(".json", "")
        )
        
        # Check if solution is correct
        is_correct = (predicted_output == expected_output)
        
        print(f"\n🎯 Results:")
        print(f"✅ Task completed successfully!")
        print(f"📊 Confidence: {confidence:.2f}")
        print(f"🎯 Correct: {'Yes' if is_correct else 'No'}")
        print(f"🔢 Token usage: {solver.get_token_counts()['total_tokens']} tokens")
        print(f"📁 Results saved to: {results_dir}")
        
        # Save final results
        final_results = {
            "task_name": task_file,
            "input_grid": input_grid,
            "expected_output": expected_output,
            "predicted_output": predicted_output,
            "confidence": confidence,
            "is_correct": is_correct,
            "token_usage": solver.get_token_counts(),
            "message_history": solver.get_message_history(),
            "intermediate_grids": solver.get_intermediate_grids()
        }
        
        with open(os.path.join(results_dir, "final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        return predicted_output, confidence, is_correct
        
    except Exception as e:
        print(f"❌ Error during solving: {str(e)}")
        return None, 0.0, False

if __name__ == "__main__":
    test_multi_agent_solver()