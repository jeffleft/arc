import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from grid_ops import GridOperations
import os
import time
from dataclasses import dataclass

@dataclass
class SolutionPlan:
    """Represents a solution plan with tool calls and reasoning."""
    description: str
    tool_sequence: List[Dict]
    confidence: float
    reasoning: str

@dataclass
class ValidationResult:
    """Represents the result of validating a solution on training examples."""
    success: bool
    accuracy: float  # Percentage of training examples that passed
    failed_examples: List[int]  # Indices of failed examples
    refinements: List[str]  # Suggested refinements
    updated_plan: Optional[SolutionPlan]

class MultiAgentARCSolver:
    def __init__(self, api_key: str, results_dir: str = None):
        """Initialize the multi-agent ARC solver with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.tools = self._define_tools()
        self.message_history = []
        self.intermediate_grids = []
        self.results_dir = results_dir
        self.current_task_dir: Optional[str] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.budget = 2000000
        
    def get_message_history(self) -> List[Dict]:
        """Get the message history for the current task."""
        return [self._message_to_dict(msg) for msg in self.message_history]
        
    def get_intermediate_grids(self) -> List[Dict]:
        """Get the sequence of intermediate grid states."""
        return self.intermediate_grids
        
    def get_token_counts(self) -> Dict[str, int]:
        """Get the total token counts for the current task."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

    def _define_tools(self) -> List[Dict]:
        """Define the available tools for grid manipulation."""
        return [
            {
                "type": "function",
                "name": "copy_grid",
                "description": "Copy the input grid to the output",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["rationale"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "copy_selection",
                "description": "Copy a selected area to one or more other places on the output grid",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_x": {"type": "integer"},
                        "start_y": {"type": "integer"},
                        "end_x": {"type": "integer"},
                        "end_y": {"type": "integer"},
                        "paste_origins": {
                            "type": "array",
                            "description": "List of positions to paste the selected area",
                            "items": {
                                "type": "array",
                                "description": "(x, y) tuples",
                                "items": {
                                    "type": "integer",
                                    "description": "Starting x or y coordinate of position"
                                }
                            }
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["start_x", "start_y", "end_x", "end_y", "paste_origins", "rationale"],
                    "additionalProperties": False
                }
            },

            {
                "type": "function",
                "name": "fill_rectangle",
                "description": "Fill a rectangle with a given color",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer"},
                        "y1": {"type": "integer"},
                        "x2": {"type": "integer"},
                        "y2": {"type": "integer"},
                        "color": {"type": "integer"},
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["x1", "y1", "x2", "y2", "color", "rationale"],
                    "additionalProperties": False
                }
            },

            {
                "type": "function",
                "name": "resize_grid",
                "description": "Resize the output grid to MxN dimensions",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "width": {
                            "type": "integer",
                            "description": "The new width of the grid",
                            "minimum": 1,
                            "maximum": 100
                        },
                        "height": {
                            "type": "integer",
                            "description": "The new height of the grid",
                            "minimum": 1,
                            "maximum": 100
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["width", "height", "rationale"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "fill_tiles",
                "description": "Fill specific tiles with given colors",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                    "color": {"type": "integer"}
                                },
                                "required": ["x", "y", "color"],
                                "additionalProperties": False
                            }
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["positions", "rationale"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "execute_python",
                "description": "Execute Python code to modify the grid. The code has access to the grid as a numpy array (self.grid), grid dimensions (self.height, self.width), and numpy (np). Example:" + \
"""
# Fill a checkerboard pattern
for y in range(self.height):
    for x in range(self.width):
        if (x + y) % 2 == 0:
            self.grid[y, x] = 1
        else:
            self.grid[y, x] = 0
""",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. The code should modify self.grid directly."
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of what the Python code does and why it's being used"
                        }
                    },
                    "required": ["code", "rationale"],
                    "additionalProperties": False
                }
            }
        ]

    def _grid_to_image(self, grid: List[List[int]]) -> str:
        """Convert a grid to a base64-encoded PNG image.
        Each tile is 16x16 pixels (matching ViT patches) with 1px white separators.
        Includes coordinate labels on the top and left sides.
        Image is padded to standard sizes: 144x144 for grids <=8x8, 256x256 for grids <=15x15, 512x512 otherwise.
        For grids larger than 512x512, the image will be truncated to show the top-left portion."""
        arr = np.array(grid, dtype=np.uint8)
        colors = {
            0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64),
            4: (255, 220, 0), 5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27),
            8: (127, 219, 255), 9: (135, 12, 37)
        }
        PATCH_SIZE = 16  # Each ViT patch is 16x16
        TILE_SIZE = 14   # Actual tile size (centered within patch)
        BORDER = 1       # Border around each tile within the patch
        LABEL_SIZE = 12
        LABEL_PADDING = 8
        height, width = arr.shape

        # Calculate required size including labels
        block_width = LABEL_SIZE + LABEL_PADDING + width * PATCH_SIZE
        block_height = LABEL_SIZE + LABEL_PADDING + height * PATCH_SIZE
        min_required = max(block_width, block_height)
        
        # Choose appropriate image size
        if min_required <= 144:
            final_size = 144
        elif min_required <= 256:
            final_size = 256
        else:
            final_size = 512

        # Create image with standard size
        img = Image.new('RGB', (final_size, final_size), (255, 255, 255))
        pixels = img.load()

        # Calculate how many tiles we can fit
        max_tiles_x = (final_size - LABEL_SIZE - LABEL_PADDING) // PATCH_SIZE
        max_tiles_y = (final_size - LABEL_SIZE - LABEL_PADDING) // PATCH_SIZE
        
        # Truncate grid dimensions if needed
        draw_width = min(width, max_tiles_x)
        draw_height = min(height, max_tiles_y)

        # Center the grid+labels block
        block_x = max(0, (final_size - block_width) // 2)
        block_y = max(0, (final_size - block_height) // 2)

        # Draw tiles
        for y in range(draw_height):
            for x in range(draw_width):
                color = colors.get(arr[y, x], (0, 0, 0))
                # Calculate patch position (each patch is 16x16)
                patch_x = block_x + LABEL_SIZE + LABEL_PADDING + x * PATCH_SIZE
                patch_y = block_y + LABEL_SIZE + LABEL_PADDING + y * PATCH_SIZE
                # Center the 14x14 tile within the 16x16 patch
                tile_x = patch_x + BORDER
                tile_y = patch_y + BORDER
                for ty in range(TILE_SIZE):
                    for tx in range(TILE_SIZE):
                        if tile_x + tx < final_size and tile_y + ty < final_size:
                            pixels[tile_x + tx, tile_y + ty] = color

        draw = ImageDraw.Draw(img)
        
        # Draw x labels (top)
        for x in range(draw_width):
            # Center label over the patch
            if x > 9:
                label_x = block_x + LABEL_SIZE + LABEL_PADDING + x * PATCH_SIZE + PATCH_SIZE // 2 - 6
            else:
                label_x = block_x + LABEL_SIZE + LABEL_PADDING + x * PATCH_SIZE + PATCH_SIZE // 2 - 2
            label_y = block_y + LABEL_PADDING // 2 + 2
            text = str(x)
            draw.text((label_x, label_y), text, fill=(0, 0, 0))
        
        # Draw y labels (left)
        for y in range(draw_height):
            if y > 9:
                label_x = block_x + LABEL_PADDING // 2
            else:
                label_x = block_x + LABEL_PADDING // 2 + 4
            # Center label next to the patch
            label_y = block_y + LABEL_SIZE + LABEL_PADDING + y * PATCH_SIZE + PATCH_SIZE // 2 - 5
            text = str(y)
            draw.text((label_x, label_y), text, fill=(0, 0, 0))

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_intermediate_state(self, step: int, grid: List[List[int]], tool: str, description: str):
        """Save an intermediate grid state in real-time."""
        if not self.current_task_dir:
            return
            
        # Create intermediate states directory if it doesn't exist
        intermediate_dir = os.path.join(self.current_task_dir, "intermediate_states")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Save grid data
        state_data = {
            "step": step,
            "grid": grid,
            "tool": tool,
            "description": description
        }
        
        # Save to states.json (append to existing or create new)
        states_file = os.path.join(intermediate_dir, "states.json")
        if os.path.exists(states_file):
            with open(states_file, "r") as f:
                states = json.load(f)
        else:
            states = []
        states.append(state_data)
        with open(states_file, "w") as f:
            json.dump(states, f, indent=2)
        
        # Save individual step files
        step_file = os.path.join(intermediate_dir, f"step_{step:03d}.json")
        with open(step_file, "w") as f:
            json.dump(state_data, f, indent=2)
            
        # Save grid image
        img_data = self._grid_to_image(grid)
        img_bytes = base64.b64decode(img_data)  # Convert base64 string to bytes
        img_path = os.path.join(intermediate_dir, f"step_{step:03d}.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
            
    def _message_to_dict(self, message):
        """Convert a message object to a JSON-serializable dictionary."""
        if hasattr(message, 'model_dump'):
            # Handle OpenAI message objects
            return message.model_dump()
        elif isinstance(message, dict):
            # Already a dictionary
            return message
        else:
            # Fallback for other types
            return str(message)
            
    def _save_message_history(self):
        """Save the current message history in real-time."""
        if not self.current_task_dir:
            return
            
        history_file = os.path.join(self.current_task_dir, "message_history.json")
        # Convert messages to JSON-serializable format
        serializable_history = [self._message_to_dict(msg) for msg in self.message_history]
        with open(history_file, "w") as f:
            json.dump(serializable_history, f, indent=2)

    def _make_api_call(self, messages: List[Dict], tools: Optional[List[Dict]] = None, max_retries: int = 3) -> Optional[object]:
        """Make an API call with retry logic."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.client.responses.create(
                    model="o4-mini",
                    input=messages,
                    tools=tools if tools else [],
                    tool_choice="auto" if tools else "none",
                    reasoning={
                        "effort": "medium",
                        "summary": "auto"
                    },
                    truncation="auto",
                )
                
                # Update token counts if available
                if hasattr(response, 'usage'):
                    print(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}, Total: {response.usage.total_tokens}")
                    self.total_input_tokens += response.usage.input_tokens
                    self.total_output_tokens += response.usage.output_tokens
                
                return response
                
            except Exception as e:
                retry_count += 1
                print(f"Error in API call (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count == max_retries:
                    print("Max retries reached.")
                    return None
                
                # Wait before retrying (exponential backoff)
                time.sleep(2 ** retry_count)
        
        return None

    def _formulate_solution(self, task_json: Dict, evolved_instructions: str) -> SolutionPlan:
        """
        Phase 1: Solution Formulation Agent
        Analyzes training examples and formulates a solution plan without executing tools.
        """
        print("🔍 Phase 1: Formulating solution plan...")
        
        # Prepare messages for solution formulation
        system_message = {
            "role": "developer",
            "content": (
                "You are a solution formulation agent for ARC puzzles. Your task is to analyze the training examples "
                "and formulate a clear solution plan WITHOUT executing any tools.\n\n"
                f"{evolved_instructions}\n\n"
                "Analyze the training examples carefully and provide:\n"
                "1. A detailed description of the pattern/rule you observe\n"
                "2. A step-by-step plan using the available tool names and parameters\n"
                "3. Your confidence level (0-10) in this approach\n"
                "4. Your reasoning process\n\n"
                "Available tools: copy_grid, copy_selection, fill_rectangle, "
                "resize_grid, fill_tiles, execute_python\n\n"
                "Format your response as a detailed analysis followed by a JSON plan with the structure:\n"
                "{\n"
                "  \"description\": \"High-level description of the solution\",\n"
                "  \"tool_sequence\": [{\"tool\": \"tool_name\", \"params\": {...}, \"rationale\": \"why\"}],\n"
                "  \"confidence\": 0.8,\n"
                "  \"reasoning\": \"Detailed reasoning\"\n"
                "}"
            )
        }

        user_message = {
            "role": "user",
            "content": []
        }
        
        # Add training examples
        for i, demo in enumerate(task_json.get("train", []), 1):
            user_message["content"].extend([
                {
                    "type": "input_text",
                    "text": f"Training Example {i}:\nInput: {json.dumps(demo['input'])}"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self._grid_to_image(demo['input'])}"
                },
                {
                    "type": "input_text", 
                    "text": f"Output: {json.dumps(demo['output'])}"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self._grid_to_image(demo['output'])}"
                }
            ])

        # Add message to history
        self.message_history.extend([system_message, user_message])
        self._save_message_history()

        # Get response
        response = self._make_api_call([system_message, user_message])
        if not response:
            return SolutionPlan("Failed to formulate solution", [], 0.0, "API call failed")

        # Process response
        plan_text = ""
        for item in response.output:
            if item.type == "reasoning":
                summary_texts = [summary.text for summary in item.summary if hasattr(summary, 'text')]
                if summary_texts:
                    reasoning_content = "Reasoning:\n" + "\n".join(summary_texts)
                    self.message_history.append({
                        "role": "assistant",
                        "content": reasoning_content
                    })
                    plan_text += reasoning_content + "\n"
            elif item.type == "message":
                self.message_history.append(item)
                plan_text += getattr(item, 'content', str(item))

        self._save_message_history()

        # Try to extract JSON plan from response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', plan_text)
            if json_match:
                plan_data = json.loads(json_match.group())
                return SolutionPlan(
                    description=plan_data.get("description", ""),
                    tool_sequence=plan_data.get("tool_sequence", []),
                    confidence=plan_data.get("confidence", 0.5),
                    reasoning=plan_data.get("reasoning", plan_text)
                )
        except Exception as e:
            print(f"Failed to parse JSON plan: {e}")

        # Fallback - return basic plan
        return SolutionPlan(
            description="Pattern analysis and transformation plan",
            tool_sequence=[],
            confidence=0.5,
            reasoning=plan_text
        )

    def _validate_on_training(self, task_json: Dict, plan: SolutionPlan) -> ValidationResult:
        """
        Phase 2: Training Validation Agent  
        Tests the solution plan on training examples and suggests refinements.
        """
        print("🧪 Phase 2: Validating solution on training examples...")
        
        training_examples = task_json.get("train", [])
        successful_tests = 0
        failed_examples = []
        
        for i, demo in enumerate(training_examples):
            input_grid = demo["input"]
            expected_output = demo["output"]
            
            # Execute the plan on this training example
            try:
                result_grid = self._execute_plan_on_grid(plan, input_grid)
                
                # Check if result matches expected output
                if result_grid and self._grids_equal(result_grid, expected_output):
                    successful_tests += 1
                    print(f"✅ Training example {i+1}: PASSED")
                else:
                    failed_examples.append(i)
                    print(f"❌ Training example {i+1}: FAILED")
                    
            except Exception as e:
                failed_examples.append(i)
                print(f"❌ Training example {i+1}: ERROR - {str(e)}")

        accuracy = successful_tests / len(training_examples) if training_examples else 0.0
        print(f"Training accuracy: {accuracy:.2%} ({successful_tests}/{len(training_examples)})")

        # If accuracy is low, try to refine the solution
        updated_plan = None
        refinements = []
        
        if accuracy < 0.8 and failed_examples:  # If less than 80% accuracy
            print("🔧 Attempting to refine solution...")
            updated_plan, refinements = self._refine_solution(task_json, plan, failed_examples)
        
        return ValidationResult(
            success=accuracy >= 0.8,
            accuracy=accuracy,
            failed_examples=failed_examples,
            refinements=refinements,
            updated_plan=updated_plan
        )

    def _execute_plan_on_grid(self, plan: SolutionPlan, input_grid: List[List[int]]) -> Optional[List[List[int]]]:
        """Execute a solution plan on a specific grid."""
        grid_ops = GridOperations(input_grid)
        
        try:
            for step in plan.tool_sequence:
                tool_name = step.get("tool")
                params = step.get("params", {})
                
                if tool_name == "copy_grid":
                    grid_ops.copy_grid()
                elif tool_name == "fill_tiles":
                    grid_ops.fill_tiles(params.get("positions", []))
                elif tool_name == "fill_rectangle":
                    grid_ops.fill_rectangle(
                        params.get("x1", 0),
                        params.get("y1", 0),
                        params.get("x2", 0),
                        params.get("y2", 0),
                        params.get("color", 0)
                    )
                elif tool_name == "resize_grid":
                    grid_ops.resize_grid(
                        params.get("width", 1),
                        params.get("height", 1)
                    )
                elif tool_name == "copy_selection":
                    grid_ops.copy_selection(
                        params.get("start_x", 0),
                        params.get("start_y", 0),
                        params.get("end_x", 0),
                        params.get("end_y", 0),
                        params.get("paste_origins", [])
                    )
                elif tool_name == "execute_python":
                    result = grid_ops.execute_python_code(params.get("code", ""))
                    if not result["success"]:
                        print(f"Python execution failed: {result['message']}")
                        return None
                        
            return grid_ops.get_grid()
            
        except Exception as e:
            print(f"Error executing plan: {str(e)}")
            return None

    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are equal."""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False
        
        for row1, row2 in zip(grid1, grid2):
            for cell1, cell2 in zip(row1, row2):
                if cell1 != cell2:
                    return False
        return True

    def _refine_solution(self, task_json: Dict, original_plan: SolutionPlan, failed_examples: List[int]) -> Tuple[Optional[SolutionPlan], List[str]]:
        """Attempt to refine the solution based on failed training examples."""
        
        # Prepare refinement message
        system_message = {
            "role": "developer", 
            "content": (
                "You are a solution refinement agent. The original solution plan failed on some training examples. "
                "Analyze the failures and provide a refined solution plan.\n\n"
                "Your task is to:\n"
                "1. Identify what went wrong with the original plan\n"
                "2. Suggest specific refinements\n"
                "3. Provide an updated solution plan in the same JSON format\n\n"
                "Focus on the failed examples and try to understand why the original approach didn't work."
            )
        }

        user_content = [
            {
                "type": "input_text",
                "text": f"Original plan:\n{json.dumps({'description': original_plan.description, 'tool_sequence': original_plan.tool_sequence, 'confidence': original_plan.confidence}, indent=2)}\n\nFailed on training examples: {failed_examples}\n\nTraining data:"
            }
        ]
        
        # Add failed training examples
        for i in failed_examples:
            if i < len(task_json.get("train", [])):
                demo = task_json["train"][i]
                user_content.extend([
                    {
                        "type": "input_text",
                        "text": f"Failed Example {i+1}:\nInput: {json.dumps(demo['input'])}"
                    },
                    {
                        "type": "input_image", 
                        "image_url": f"data:image/png;base64,{self._grid_to_image(demo['input'])}"
                    },
                    {
                        "type": "input_text",
                        "text": f"Expected Output: {json.dumps(demo['output'])}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{self._grid_to_image(demo['output'])}"
                    }
                ])

        user_message = {"role": "user", "content": user_content}
        
        # Get refinement response
        response = self._make_api_call([system_message, user_message])
        if not response:
            return None, ["Failed to get refinement suggestions"]

        # Process response
        refinement_text = ""
        for item in response.output:
            if item.type == "reasoning":
                summary_texts = [summary.text for summary in item.summary if hasattr(summary, 'text')]
                if summary_texts:
                    refinement_text += "\n".join(summary_texts) + "\n"
            elif item.type == "message":
                refinement_text += getattr(item, 'content', str(item))

        # Try to extract refined plan
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', refinement_text)
            if json_match:
                plan_data = json.loads(json_match.group())
                refined_plan = SolutionPlan(
                    description=plan_data.get("description", original_plan.description),
                    tool_sequence=plan_data.get("tool_sequence", original_plan.tool_sequence),
                    confidence=plan_data.get("confidence", original_plan.confidence * 0.8),
                    reasoning=plan_data.get("reasoning", refinement_text)
                )
                return refined_plan, [refinement_text]
        except Exception as e:
            print(f"Failed to parse refined plan: {e}")

        return None, [refinement_text]

    def _apply_solution(self, input_grid: List[List[int]], plan: SolutionPlan) -> Tuple[List[List[int]], float]:
        """
        Phase 3: Application Agent
        Applies the validated solution plan to the input grid.
        """
        print("🎯 Phase 3: Applying solution to input grid...")
        
        # Initialize grid operations
        grid_ops = GridOperations(input_grid)
        
        # Save initial state
        initial_state = {
            "step": 0,
            "grid": grid_ops.get_grid(),
            "tool": "initial",
            "description": "Initial grid state - Application Phase"
        }
        self.intermediate_grids.append(initial_state)
        self._save_intermediate_state(**initial_state)
        
        step = 1
        
        try:
            # Execute each step in the plan
            for i, tool_step in enumerate(plan.tool_sequence):
                tool_name = tool_step.get("tool")
                params = tool_step.get("params", {})
                rationale = tool_step.get("rationale", f"Step {i+1} of solution plan")
                
                print(f"  Executing step {i+1}: {tool_name}")
                
                # Execute the tool
                if tool_name == "copy_grid":
                    grid_ops.copy_grid()
                elif tool_name == "fill_tiles":
                    grid_ops.fill_tiles(params.get("positions", []))
                elif tool_name == "fill_rectangle":
                    grid_ops.fill_rectangle(
                        params.get("x1", 0),
                        params.get("y1", 0),
                        params.get("x2", 0),
                        params.get("y2", 0),
                        params.get("color", 0)
                    )
                elif tool_name == "resize_grid":
                    grid_ops.resize_grid(
                        params.get("width", 1),
                        params.get("height", 1)
                    )
                elif tool_name == "copy_selection":
                    grid_ops.copy_selection(
                        params.get("start_x", 0),
                        params.get("start_y", 0),
                        params.get("end_x", 0),
                        params.get("end_y", 0),
                        params.get("paste_origins", [])
                    )
                elif tool_name == "execute_python":
                    result = grid_ops.execute_python_code(params.get("code", ""))
                    if not result["success"]:
                        print(f"  ❌ Python execution failed: {result['message']}")
                        break
                else:
                    print(f"  ⚠️  Unknown tool: {tool_name}")
                    continue
                
                # Save intermediate state
                current_grid = grid_ops.get_grid()
                state = {
                    "step": step,
                    "grid": current_grid,
                    "tool": tool_name,
                    "description": f"Applied {tool_name}: {rationale}"
                }
                self.intermediate_grids.append(state)
                self._save_intermediate_state(**state)
                step += 1
                
                print(f"  ✅ Step {i+1} completed")
            
            final_grid = grid_ops.get_grid()
            print(f"✅ Phase 3 completed: Solution applied to input grid")
            
            return final_grid, plan.confidence
            
        except Exception as e:
            print(f"❌ Error in application phase: {str(e)}")
            return grid_ops.get_grid(), 0.1

    def _verify_solution(self, input_grid: List[List[int]], output_grid: List[List[int]], plan: SolutionPlan) -> Tuple[List[List[int]], float]:
        """
        Phase 4: Verification Agent
        Performs a final double-check of the solution.
        """
        print("🔍 Phase 4: Verifying solution...")
        
        # Prepare verification message
        system_message = {
            "role": "developer",
            "content": (
                "You are a solution verification agent. Your task is to analyze the final output and determine if it "
                "looks correct based on the original plan and the input grid.\n\n"
                "Provide:\n"
                "1. Your assessment of whether the solution looks correct\n"
                "2. Any issues you notice\n"
                "3. A confidence score (0-10) for the solution\n"
                "4. If you notice issues, suggest a quick fix if possible\n\n"
                "Be thorough but concise in your analysis."
            )
        }

        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Original plan: {plan.description}\n\nInput grid: {json.dumps(input_grid)}"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self._grid_to_image(input_grid)}"
                },
                {
                    "type": "input_text", 
                    "text": f"Output grid: {json.dumps(output_grid)}"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self._grid_to_image(output_grid)}"
                },
                {
                    "type": "input_text",
                    "text": "Please verify if this output looks correct based on the transformation plan."
                }
            ]
        }

        # Add to message history
        self.message_history.extend([system_message, user_message])
        self._save_message_history()

        # Get verification response
        response = self._make_api_call([system_message, user_message])
        if not response:
            print("⚠️  Could not get verification response, returning original solution")
            return output_grid, plan.confidence * 0.8

        # Process verification response
        verification_text = ""
        for item in response.output:
            if item.type == "reasoning":
                summary_texts = [summary.text for summary in item.summary if hasattr(summary, 'text')]
                if summary_texts:
                    reasoning_content = "Verification Reasoning:\n" + "\n".join(summary_texts)
                    self.message_history.append({
                        "role": "assistant",
                        "content": reasoning_content
                    })
                    verification_text += reasoning_content + "\n"
            elif item.type == "message":
                self.message_history.append(item)
                verification_text += getattr(item, 'content', str(item))

        self._save_message_history()

        # Try to extract confidence score from verification
        confidence = plan.confidence
        try:
            import re
            # Look for confidence scores in various formats
            confidence_patterns = [
                r'confidence[:\s]+(\d+(?:\.\d+)?)',
                r'score[:\s]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*[/\\]\s*10',
                r'(\d+(?:\.\d+)?)(?:\s*out\s*of\s*10)?'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, verification_text.lower())
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-1 range if it seems to be 0-10
                    if score > 1:
                        score = score / 10.0
                    confidence = score
                    break
        except:
            pass

        print(f"✅ Phase 4 completed: Verification confidence = {confidence:.2f}")
        print(f"Verification notes: {verification_text[:200]}...")
        
        return output_grid, confidence

    def solve_task(self, task_json: Dict, input_grid: List[List[int]], evolved_instructions: str, task_name: str) -> Tuple[List[List[int]], float]:
        """
        Main solver method implementing the multi-agent approach.
        
        Returns:
            Tuple of (output_grid, confidence_score)
        """
        print(f"\n🚀 Starting multi-agent solver for task: {task_name}")
        
        # Reset state for new task
        self.message_history = []
        self.intermediate_grids = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Set up task directory
        if self.results_dir:
            self.current_task_dir = os.path.join(self.results_dir, task_name)
            os.makedirs(self.current_task_dir, exist_ok=True)

        try:
            # Phase 1: Solution Formulation
            solution_plan = self._formulate_solution(task_json, evolved_instructions)
            print(f"📋 Solution plan: {solution_plan.description}")
            print(f"🎯 Initial confidence: {solution_plan.confidence:.2f}")
            
            # Phase 2: Training Validation
            validation_result = self._validate_on_training(task_json, solution_plan)
            
            # Use refined plan if available and validation improved
            final_plan = solution_plan
            if validation_result.updated_plan and validation_result.updated_plan.confidence > solution_plan.confidence:
                final_plan = validation_result.updated_plan
                print(f"🔧 Using refined plan with confidence: {final_plan.confidence:.2f}")
            
            # Check if we should proceed with low validation accuracy
            if not validation_result.success:
                print(f"⚠️  Low training accuracy ({validation_result.accuracy:.2%}), proceeding with caution...")
            
            # Phase 3: Application 
            output_grid, application_confidence = self._apply_solution(input_grid, final_plan)
            
            # Phase 4: Verification
            final_grid, final_confidence = self._verify_solution(input_grid, output_grid, final_plan)
            
            print(f"✅ Multi-agent solver completed!")
            print(f"📊 Final confidence: {final_confidence:.2f}")
            print(f"🔢 Token usage: {self.total_input_tokens + self.total_output_tokens} total tokens")
            
            return final_grid, final_confidence

        except Exception as e:
            print(f"❌ Error in multi-agent solver: {str(e)}")
            # Return the input grid as fallback
            return input_grid, 0.0