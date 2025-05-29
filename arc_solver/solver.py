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

class ARCSolver:
    def __init__(self, api_key: str, results_dir: str = None):
        """Initialize the ARC solver with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.tools = self._define_tools()
        self.message_history = []
        self.intermediate_grids = []
        self.results_dir = results_dir
        self.current_task_dir = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
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
                "name": "fill_pattern",
                "description": "Fill tiles in a pattern with fixed interval and direction",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_x": {"type": "integer"},
                        "start_y": {"type": "integer"},
                        "direction": {"type": "string", "enum": ["horizontal", "vertical"]},
                        "interval": {"type": "integer"},
                        "color": {"type": "integer"},
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["start_x", "start_y", "direction", "interval", "color", "rationale"],
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
                "name": "translate",
                "description": "Translate the grid by a given offset",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dx": {"type": "integer"},
                        "dy": {"type": "integer"},
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["dx", "dy", "rationale"],
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
                "name": "finish",
                "description": "Indicate that the solution is complete",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confidence": {
                            "type": "integer",
                            "description": "Confidence score from 0-10 in the solution's correctness",
                            "minimum": 0,
                            "maximum": 10
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why the solution is complete and why the confidence score was chosen"
                        }
                    },
                    "required": ["confidence", "rationale"],
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
        Image is padded to standard sizes: 144x144 for grids <=8x8, 256x256 for grids <=15x15, 512x512 otherwise."""
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
        else:  # For larger grids, scale down
            final_size = 512
            # Calculate scaling factor to fit the grid
            scale = min(512 / block_width, 512 / block_height)
            # Scale down the grid
            new_height = int(height * scale)
            new_width = int(width * scale)
            # Use nearest neighbor interpolation to preserve colors
            arr = np.array(Image.fromarray(arr).resize((new_width, new_height), Image.NEAREST))
            height, width = arr.shape
            # Recalculate block dimensions
            block_width = LABEL_SIZE + LABEL_PADDING + width * PATCH_SIZE
            block_height = LABEL_SIZE + LABEL_PADDING + height * PATCH_SIZE

        img = Image.new('RGB', (final_size, final_size), (255, 255, 255))
        pixels = img.load()

        # Center the grid+labels block
        block_x = max(0, (final_size - block_width) // 2)
        block_y = max(0, (final_size - block_height) // 2)

        # Draw tiles
        for y in range(height):
            for x in range(width):
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
        for x in range(width):
            # Center label over the patch
            if x > 9:
                label_x = block_x + LABEL_SIZE + LABEL_PADDING + x * PATCH_SIZE + PATCH_SIZE // 2 - 6
            else:
                label_x = block_x + LABEL_SIZE + LABEL_PADDING + x * PATCH_SIZE + PATCH_SIZE // 2 - 2
            label_y = block_y + LABEL_PADDING // 2 + 2
            text = str(x)
            draw.text((label_x, label_y), text, fill=(0, 0, 0))
        
        # Draw y labels (left)
        for y in range(height):
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
        
    def solve_task(self, task_json: Dict, input_grid: List[List[int]], evolved_instructions: str, task_name: str) -> List[List[int]]:
        """Solve an ARC task given the task JSON and input grid."""
        # Reset message history and intermediate grids for new task
        self.message_history = []
        self.intermediate_grids = []
        
        # Set up task directory
        if self.results_dir:
            self.current_task_dir = os.path.join(self.results_dir, task_name)
            os.makedirs(self.current_task_dir, exist_ok=True)
        
        # Initialize grid operations
        grid_ops = GridOperations(input_grid)
        
        # Save initial grid state
        initial_state = {
            "step": 0,
            "grid": grid_ops.get_grid(),
            "tool": "initial",
            "description": "Initial grid state"
        }
        self.intermediate_grids.append(initial_state)
        self._save_intermediate_state(**initial_state)
        
        # Prepare the system message with tools
        system_message = {
            "role": "developer",
            "content": ("You are an ARC grid-puzzle solver. You will be given task demonstrations, from which you can infer "
                         "rules and patterns that define the task. You will then be given an input grid and tools to generate "
                         "the solution/output grid.\n"
                         f"{evolved_instructions}"
                         "\n\nFirst, return your reasoning about what the underlying rules and possible solution should be. "
                         "Next, execute the tool calls to generate the output grid (you will never output the grid directly!) Once each tool call returns, the user will send the "
                         "updated representation of the output grid. Continue with the tool calls until you are confident in your "
                         "solution, then use the finish tool with a confidence score.")
        }
        
        # Add system message to log
        self.message_history.append(system_message)
        self._save_message_history()
        
        # Prepare the user message with context
        user_message = {
            "role": "user",
            "content": []
        }
        
        # Add training examples sequentially
        for i, demo in enumerate(task_json.get("train", []), 1):
            # Add input
            user_message["content"].append({
                "type": "input_text",
                "text": f"Training input {i}:\n{json.dumps(demo['input'])}"
            })
            user_message["content"].append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{self._grid_to_image(demo['input'])}"
            })
            
            # Add output
            user_message["content"].append({
                "type": "input_text",
                "text": f"Training output {i}:\n{json.dumps(demo['output'])}"
            })
            user_message["content"].append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{self._grid_to_image(demo['output'])}"
            })
        
        # Add the current input grid
        user_message["content"].append({
            "type": "input_text",
            "text": f"\nTask input grid:\n{json.dumps(input_grid)}"
        })
        user_message["content"].append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{self._grid_to_image(input_grid)}"
        })
        
        # Add user message to log
        self.message_history.append(user_message)
        self._save_message_history()
        
        # Initialize messages list to pass to gpt
        messages = [system_message, user_message]
        
        # Loop until finish tool is called
        step = 1
        previous_response_id = None
        while True:
            # Get completion from responses API
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = self.client.responses.create(
                        model="o4-mini",
                        input=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        reasoning={
                            "effort": "medium",
                            "summary": "auto"
                        },
                        previous_response_id=previous_response_id
                    )
                    
                    # Print token counts if available
                    if hasattr(response, 'usage'):
                        print(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}, Total: {response.usage.total_tokens}")
                        if hasattr(response.usage, 'input_tokens_details'):
                            print(f"Input tokens details - Cached: {response.usage.input_tokens_details.cached_tokens}")
                        if hasattr(response.usage, 'output_tokens_details'):
                            print(f"Output tokens details - Reasoning: {response.usage.output_tokens_details.reasoning_tokens}")
                        
                        # Update total token counts
                        self.total_input_tokens += response.usage.input_tokens
                        self.total_output_tokens += response.usage.output_tokens
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error in API call (attempt {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count == max_retries:
                        print("Max retries reached. Skipping this task.")
                        return None, 0  # Return None grid and 0 confidence to indicate task should be skipped
                    
                    # Wait before retrying (exponential backoff)
                    time.sleep(2 ** retry_count)
            
            # Store the response ID for the next iteration
            previous_response_id = response.id

            # Check if reponse does not have either a tool call or reasoning, if so, skip rendering grid
            if not any(item.type == "function_call" or item.type == "reasoning" for item in response.output):
                print("Response does not have either a tool call or reasoning!")
                for item in response.output:
                    self.message_history.append(item)
                self._save_message_history()
                continue

            # Process all items in the response output
            for item in response.output:
                if item.type == "reasoning":
                    # Add reasoning to message log
                    summary_texts = [summary.text for summary in item.summary]
                    if summary_texts:  # Only append if there are actual summary texts
                        self.message_history.append({
                            "role": "assistant",
                            "content": "Reasoning:\n" + "\n".join(summary_texts)
                        })
                        self._save_message_history()
                elif item.type == "function_call":
                    # Add the function call to messages
                    function_call = {
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments
                    }
                    messages.append(function_call)

                    # log the function call
                    self.message_history.append(function_call)
                    self._save_message_history()
                    
                    function_name = item.name
                    function_args = json.loads(item.arguments)
                    
                    # Apply the appropriate tool operation
                    if function_name == "fill_tiles":
                        grid_ops.fill_tiles(function_args["positions"])
                    elif function_name == "copy_grid":
                        grid_ops.copy_grid()
                    elif function_name == "fill_pattern":
                        grid_ops.fill_pattern(
                            function_args["start_x"],
                            function_args["start_y"],
                            function_args["direction"],
                            function_args["interval"],
                            function_args["color"]
                        )
                    elif function_name == "fill_rectangle":
                        grid_ops.fill_rectangle(
                            function_args["x1"],
                            function_args["y1"],
                            function_args["x2"],
                            function_args["y2"],
                            function_args["color"]
                        )
                    elif function_name == "translate":
                        grid_ops.translate(
                            function_args["dx"],
                            function_args["dy"]
                        )
                    elif function_name == "resize_grid":
                        grid_ops.resize_grid(function_args["width"], function_args["height"])
                    elif function_name == "copy_selection":
                        grid_ops.copy_selection(
                            function_args["start_x"],
                            function_args["start_y"],
                            function_args["end_x"],
                            function_args["end_y"],
                            function_args["paste_origins"]
                        )
                    elif function_name == "execute_python":
                        result = grid_ops.execute_python_code(function_args["code"])
                        if not result["success"]:
                            # If there was an error, add it to messages and continue
                            tool_response = {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": result["message"]
                            }
                            messages.append(tool_response)
                            self.message_history.append(tool_response)
                            self._save_message_history()
                            continue
                    elif function_name == "finish":
                        current_grid = grid_ops.get_grid()
                        current_image = self._grid_to_image(current_grid)
                        return current_grid, function_args["confidence"]
                    
                    # Add tool response to messages
                    tool_response = {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"status": "success"})
                    }
                    messages.append(tool_response)

                    # log the tool response
                    self.message_history.append(tool_response)
                    self._save_message_history()
                else:
                    # eg. type == "message"
                    self.message_history.append(item)
                    self._save_message_history()
            
            # Save intermediate grid state after all tool calls
            current_grid = grid_ops.get_grid()
            function_calls = [item for item in response.output if item.type == "function_call"]
            state = {
                "step": step,
                "grid": current_grid,
                "tool": "multiple" if len(function_calls) > 1 else function_calls[0].name if function_calls else "none",
                "description": ", ".join([item.name for item in function_calls])
            }
            self.intermediate_grids.append(state)
            self._save_intermediate_state(**state)
            step += 1

            # Remove last grid update from messages if exists
            for m in reversed(messages):
                if m.get("role") == "user" and m.get("content")[0].get("type") == "input_text" and m.get("content")[0].get("text").startswith("Current output grid state:"):
                    messages.remove(m)
                    break
            
            # Add the current grid state to the message history
            current_image = self._grid_to_image(current_grid)
            grid_update = {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Current output grid state:\n{json.dumps(current_grid)}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{current_image}"
                    },
                    {
                        "type": "input_text",
                        "text": "Is this what you expected?"
                    }
                ]
            }
            messages.append(grid_update) 

            # log the grid update
            self.message_history.append(grid_update)
            self._save_message_history()