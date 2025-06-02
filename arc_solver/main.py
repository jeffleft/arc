import os
import json
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from solver import ARCSolver
# from grid_ops import GridOperations # No longer directly used in main
from prompt_evolution import PromptEvolution
# import shutil # No longer directly used in main
from datetime import datetime
import base64
import concurrent.futures
import traceback


def load_task(task_path: str) -> Dict:
    """Load a task JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def save_results(task_name: str, results_dir: str, task_results: Dict):
    """Save task results to the results directory."""
    # Create task directory
    task_dir = os.path.join(results_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # task_results should arrive here already prepared for serialization,
    # with image data as base64 strings and no live solver instances.
    serializable_results = task_results

    # Save task results (should be directly serializable)
    with open(os.path.join(task_dir, "results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Save images
    if "images" in serializable_results and serializable_results["images"]:
        for img_name, img_data in serializable_results["images"].items():
            img_path = os.path.join(task_dir, f"{img_name}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_data))

    # Save message history
    if "message_history" in serializable_results and serializable_results["message_history"]:
        with open(os.path.join(task_dir, "message_history.json"), "w") as f:
            json.dump(serializable_results["message_history"], f, indent=2)

    # Save intermediate grid states
    if "intermediate_states" in serializable_results and serializable_results["intermediate_states"]:
        intermediate_dir = os.path.join(task_dir, "intermediate_states")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Save intermediate states info
        with open(os.path.join(intermediate_dir, "states.json"), "w") as f:
            json.dump(serializable_results["intermediate_states"], f, indent=2)
        
        # Save intermediate grid images (using pre-generated base64 data)
        if "intermediate_states_images" in serializable_results and serializable_results["intermediate_states_images"]:
            for img_info in serializable_results["intermediate_states_images"]:
                step = img_info["step"]
                img_data_b64 = img_info["image_base64"]
                img_path = os.path.join(intermediate_dir, f"step_{step:03d}.png")
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(img_data_b64))
            # Grid data for each step is already saved in states.json (metadata part of intermediate_states)
        # else:
            # print(f"No intermediate_states_images found for {task_name} or it's empty.")


def evaluate_solution(predicted: List[List[int]], expected: List[List[int]], confidence: int, plan: str, solver: ARCSolver, api_key: str) -> Tuple[int, str]:
    """Evaluate the solution accuracy and get LLM commentary.
    Returns (score, commentary) where score is 0 or 1."""

    # Re-initialize a client for evaluation if needed, or pass one.
    # For simplicity, ARCSolver can create its own client.
    # If solver is None (e.g. due to an error in task processing), we can't evaluate.
    if solver is None:
        return 0, "Cannot evaluate solution: Solver instance is missing."

    if not predicted: # Handle cases where predicted_output might be None or empty
        return 0, "Predicted output is missing or empty."

    if len(predicted) != len(expected) or (expected and len(predicted[0]) != len(expected[0])):
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
    # The solver passed here should be the one from process_task, which has its own client.

    # Load evaluation system prompt from file
    eval_prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "evaluation_prompt.txt")
    try:
        with open(eval_prompt_path, 'r') as f:
            evaluation_system_prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: Evaluation prompt file not found at {eval_prompt_path}. Using fallback prompt.")
        evaluation_system_prompt_content = "You are an expert at analyzing ARC puzzle solutions. Your task is to explain why a solution is correct or incorrect, focusing on the underlying rules inferred. No need to directly output any grids; just be descriptive. Keep it concise without missing any details. Skip the formatting/markdown."
    except Exception as e:
        print(f"Error reading evaluation prompt file {eval_prompt_path}: {e}. Using fallback prompt.")
        evaluation_system_prompt_content = "You are an expert at analyzing ARC puzzle solutions. Your task is to explain why a solution is correct or incorrect, focusing on the underlying rules inferred. No need to directly output any grids; just be descriptive. Keep it concise without missing any details. Skip the formatting/markdown."

    response = solver.client.chat.completions.create(
        model="gpt-4o", # Using a slightly cheaper/faster model for evaluation
        # max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": evaluation_system_prompt_content
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
4. How the model went wrong/right with its plan

Finally, also output a numeric score between 0 and 10 for the model's approach."""
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
    results_dir = os.path.join("results", f"eval_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize components (PromptEvolution for initial prompt, others initialized per task)
    prompt_evolution = PromptEvolution(api_key) # Keep for initial prompt generation
    initial_prompt = prompt_evolution._get_initial_prompt() # Get initial prompt once

    # Save initial prompt
    with open(os.path.join(results_dir, "initial_prompt.txt"), "w") as f:
        f.write(initial_prompt)

    # Initialize token usage tracking
    total_token_usage = {
        "tasks": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0
    }

    # Load tasks
    data_dir = "../data/v2"
    training_dir = os.path.join(data_dir, "evaluation")
    
    task_files = [f for f in os.listdir(training_dir) if f.endswith(".json")]

    # Apply task limit if needed (e.g. for testing)
    # task_limit = 5 # Uncomment and set to desired number to limit tasks
    # task_files_to_process = task_files[:task_limit]

    # Example: Process tasks from index 50 to 54 (total 5 tasks)
    start_index = 0 # Change to 50 for original behavior
    end_index = 5 # Change to 55 for original behavior
    task_files_to_process = task_files[start_index:end_index]


    futures = []
    # Max workers can be adjusted based on CPU cores and task nature (IO bound vs CPU bound)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for task_filename in task_files_to_process:
            task_file_path = os.path.join(training_dir, task_filename)
            futures.append(executor.submit(process_task, task_file_path, results_dir, api_key, initial_prompt, task_filename))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result["status"] == "success":
                    task_name_json = result["task_name"] # This is task_filename
                    task_name_no_ext = task_name_json.replace(".json", "")

                    save_results(task_name_no_ext, results_dir, result["data"])

                    # Update total token usage
                    token_counts = result["data"]["token_usage"]
                    total_token_usage["tasks"].append({
                        "task_name": task_name_json,
                        "token_usage": token_counts
                    })
                    total_token_usage["total_input_tokens"] += token_counts["input_tokens"]
                    total_token_usage["total_output_tokens"] += token_counts["output_tokens"]
                    total_token_usage["total_tokens"] += token_counts["total_tokens"]

                    print(f"Task: {task_name_json}")
                    print(f"Score: {result['data']['score']}")
                    print(f"Confidence: {result['data']['confidence']:.2f}")
                    print("\nAnalysis:")
                    print(result['data']['commentary'])
                    print("---")
                else:
                    print(f"Task {result.get('task_name', 'Unknown')} failed: {result.get('error_message', 'Unknown error')}")
                    if 'traceback' in result:
                        print(result['traceback'])
                    # Optionally save error info to a file
                    error_info = {
                        "task_name": result.get('task_name', 'Unknown'),
                        "error_message": result.get('error_message', 'Unknown error'),
                        "traceback": result.get('traceback', '')
                    }
                    with open(os.path.join(results_dir, f"error_{result.get('task_name', 'unknown_task').replace('.json','')}.json"), "w") as f:
                        json.dump(error_info, f, indent=2)

            except Exception as exc:
                # This catches exceptions from future.result() itself (e.g., if the process died)
                task_name_future = "Unknown (error during future processing)"
                # Attempt to find which task failed if possible, though future might not be easily identifiable here
                # For now, log a generic error.
                print(f"A task generated an exception during future processing: {exc}")
                traceback.print_exc()
                # Save error info
                error_info = {
                    "task_name": task_name_future,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc()
                }
                with open(os.path.join(results_dir, f"error_{task_name_future.replace(' ','_')}.json"), "w") as f:
                    json.dump(error_info, f, indent=2)

    # Save final prompt evolution history (if re-enabled)
    # with open(os.path.join(results_dir, "prompt_evolution_history.json"), "w") as f:
    #     json.dump(prompt_evolution.prompt_history, f, indent=2)
        
    # Save token usage summary
    with open(os.path.join(results_dir, "token_usage_summary.json"), "w") as f:
        json.dump(total_token_usage, f, indent=2)

    print("\nToken Usage Summary:")
    print(f"Total Input Tokens: {total_token_usage['total_input_tokens']}")
    print(f"Total Output Tokens: {total_token_usage['total_output_tokens']}")
    print(f"Total Tokens: {total_token_usage['total_tokens']}")

def process_task(task_file_path: str, results_dir: str, api_key: str, current_prompt: str, task_filename: str) -> Dict:
    """
    Processes a single ARC task. This function is designed to be run in a separate process.
    """
    try:
        # Each process needs its own solver instance
        solver = ARCSolver(api_key, results_dir) # results_dir might not be strictly needed by solver itself if not writing from it
        
        task_data = load_task(task_file_path)
        
        test_case = task_data["test"][0]
        input_grid = test_case["input"]
        expected_output = test_case["output"]
        
        predicted_output, confidence = solver.solve_task(
            task_data,
            input_grid,
            current_prompt,
            task_filename.replace(".json", "")
        )

        if predicted_output is None:
            # Task failed in solving process
            score = 0
            commentary = "Task failed due to an error or timeout in the solving process."
            # Use input_grid as a fallback for predicted_output to avoid errors in image generation
            # though it might be better to have specific error handling for images too.
            final_predicted_output = input_grid
            confidence = 0 # Ensure confidence is defined
        else:
            final_predicted_output = predicted_output
            try:
                plan = [x for x in solver.get_message_history() if x.get("role") == "assistant"][0]["content"]
            except IndexError:
                plan = "Plan not found in message history."
            
            # Evaluate the solution - pass api_key for evaluate_solution to potentially use its own client
            score, commentary = evaluate_solution(final_predicted_output, expected_output, confidence, plan, solver, api_key)

        # Prepare task results for this process
        # The main solver instance (solver) is needed by save_results for image generation.
        task_results_data = {
            "task_name": task_filename, # Use the original filename passed as arg
            "input_grid": input_grid,
            "expected_output": expected_output,
            "predicted_output": final_predicted_output,
            "score": score,
            "confidence": confidence,
            "commentary": commentary,
            "prompt": current_prompt, # The prompt used for this task
            "images": { # Base64 encoded images
                "input": solver._grid_to_image(input_grid),
                "expected": solver._grid_to_image(expected_output),
                "predicted": solver._grid_to_image(final_predicted_output if final_predicted_output else [])
            },
            "message_history": solver.get_message_history(),
            "intermediate_states": solver.get_intermediate_grids(), # This is the metadata
            "intermediate_states_images": [ # Base64 encoded images for intermediate steps
                {"step": state["step"], "image_base64": solver._grid_to_image(state["grid"])}
                for state in solver.get_intermediate_grids()
            ],
            # "solver": solver, # DO NOT PASS SOLVER INSTANCE, IT'S NOT PICKLABLE
            "token_usage": solver.get_token_counts()
        }
        return {
            "status": "success",
            "task_name": task_filename,
            "data": task_results_data
        }

    except Exception as e:
        print(f"Error processing task {task_filename}: {e}")
        # It's crucial that this function returns a dictionary, even on error,
        # so the main thread can handle it.
        return {
            "status": "error",
            "task_name": task_filename,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    main() 