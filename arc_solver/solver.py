import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from grid_ops import GridOperations
import os

class ARCSolver:
    def __init__(self, api_key: str, results_dir: str = None):
        """Initialize the ARC solver with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.tools = self._define_tools()
        self.message_history = []
        self.intermediate_grids = []
        self.results_dir = results_dir
        self.current_task_dir = None
        
    def get_message_history(self) -> List[Dict]:
        """Get the message history for the current task."""
        return [self._message_to_dict(msg) for msg in self.message_history]
        
    def get_intermediate_grids(self) -> List[Dict]:
        """Get the sequence of intermediate grid states."""
        return self.intermediate_grids
        
    def _define_tools(self) -> List[Dict]:
        """Define the available tools for grid manipulation."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "copy_grid",
                    "description": "Copy the input grid to the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rationale": {
                                "type": "string",
                                "description": "Explanation of why this operation is being performed"
                            }
                        },
                        "required": ["rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_selection",
                    "description": "Copy a selected area to one or more other places on the output grid",
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
                        "required": ["start_x", "start_y", "end_x", "end_y", "paste_origins", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fill_pattern",
                    "description": "Fill tiles in a pattern with fixed interval and direction",
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
                        "required": ["start_x", "start_y", "direction", "interval", "color", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fill_rectangle",
                    "description": "Fill a rectangle with a given color",
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
                        "required": ["x1", "y1", "x2", "y2", "color", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "translate",
                    "description": "Translate the grid by a given offset",
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
                        "required": ["dx", "dy", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "resize_grid",
                    "description": "Resize the output grid to MxN dimensions",
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
                        "required": ["width", "height", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fill_tiles",
                    "description": "Fill specific tiles with given colors",
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
                                    }
                                }
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Explanation of why this operation is being performed"
                            }
                        },
                        "required": ["positions", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Indicate that the solution is complete",
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
                        "required": ["confidence", "rationale"]
                    }
                }
            }
        ]

    def _grid_to_image(self, grid: List[List[int]]) -> str:
        """Convert a grid to a base64-encoded PNG image.
        Each tile is 8x8 pixels with 1px white separators.
        Includes coordinate labels on the top and left sides."""
        # Convert grid to numpy array
        arr = np.array(grid, dtype=np.uint8)
        
        # Create a color mapping matching ARC's official colors
        colors = {
            0: (0, 0, 0),           # Black
            1: (0, 116, 217),       # Blue (#0074D9)
            2: (255, 65, 54),       # Red (#FF4136)
            3: (46, 204, 64),       # Green (#2ECC40)
            4: (255, 220, 0),       # Yellow (#FFDC00)
            5: (170, 170, 170),     # Grey (#AAAAAA)
            6: (240, 18, 190),      # Fuschia (#F012BE)
            7: (255, 133, 27),      # Orange (#FF851B)
            8: (127, 219, 255),     # Teal (#7FDBFF)
            9: (135, 12, 37)        # Brown (#870C25)
        }
        
        # Constants for rendering
        TILE_SIZE = 13
        SEPARATOR_WIDTH = 3
        LABEL_SIZE = 12  # Size for coordinate labels
        LABEL_PADDING = 2  # Padding around labels
        
        # Calculate image dimensions including labels
        height, width = arr.shape
        img_width = width * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH + LABEL_SIZE + LABEL_PADDING
        img_height = height * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH + LABEL_SIZE + LABEL_PADDING
        
        # Create RGB image
        img = Image.new('RGB', (img_width, img_height), (255, 255, 255))  # White background
        pixels = img.load()
        
        # Draw tiles
        for y in range(height):
            for x in range(width):
                color = colors.get(arr[y, x], (0, 0, 0))
                
                # Calculate tile position (offset by label size)
                tile_x = x * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH + LABEL_SIZE + LABEL_PADDING
                tile_y = y * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH + LABEL_SIZE + LABEL_PADDING
                
                # Fill tile
                for ty in range(TILE_SIZE):
                    for tx in range(TILE_SIZE):
                        pixels[tile_x + tx, tile_y + ty] = color
        
        # Draw coordinate labels
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # Draw x-coordinate labels (top)
        for x in range(width):
            label_x = x * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH + LABEL_SIZE + LABEL_PADDING + TILE_SIZE//2
            label_y = LABEL_PADDING*2
            draw.text((label_x, label_y), str(x), fill=(0, 0, 0), font=font, anchor="mm")
        
        # Draw y-coordinate labels (left)
        for y in range(height):
            label_x = LABEL_PADDING*3
            label_y = y * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH + LABEL_SIZE + LABEL_PADDING + TILE_SIZE//2
            draw.text((label_x, label_y), str(y), fill=(0, 0, 0), font=font, anchor="mm")
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_intermediate_state(self, step: int, grid: List[List[int]], tool: str, description: str, rationale: str):
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
            "description": description,
            "rationale": rationale
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
            "description": "Initial grid state",
            "rationale": "Starting state before any transformations"
        }
        self.intermediate_grids.append(initial_state)
        self._save_intermediate_state(**initial_state)
        
        # Prepare the system message with tools
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": ("You are an ARC grid-puzzle solver. You will be given task demonstrations, from which you can infer "
                             "rules and patterns that define the task. You will then be given an input grid and tools to generate "
                             "the solution/output grid.\n"
                             f"{evolved_instructions}"
                             "\n\nFirst, return your reasoning about what the underlying rules and possible solution should be. "
                             "Next, execute the tool calls to generate the output grid (do not output the grid directly). Once each tool call returns, the user will send the "
                             "updated representation of the output grid. Continue with the tool calls until you are confident in your "
                             "solution, then use the finish tool with a confidence score.")
                }
            ]
        }
        
        # Add system message to history
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
                "type": "text",
                "text": f"Sample input {i}:\n{json.dumps(demo['input'])}"
            })
            user_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._grid_to_image(demo['input'])}"
                }
            })
            
            # Add output
            user_message["content"].append({
                "type": "text",
                "text": f"Sample output {i}:\n{json.dumps(demo['output'])}"
            })
            user_message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._grid_to_image(demo['output'])}"
                }
            })
        
        # Add the current input grid
        user_message["content"].append({
            "type": "text",
            "text": f"\nTask input grid:\n{json.dumps(input_grid)}"
        })
        user_message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{self._grid_to_image(input_grid)}"
            }
        })
        
        # Add user message to history
        self.message_history.append(user_message)
        self._save_message_history()
        
        # Initialize messages list for conversation history
        messages = [system_message, user_message]
        
        # Loop until finish tool is called
        step = 1
        while True:
            # Get completion from GPT-4
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            # Add the response to message history
            assistant_message = response.choices[0].message
            self.message_history.append(assistant_message)
            self._save_message_history()
            messages.append(assistant_message)

            # Check if there are no tool calls
            if not assistant_message.tool_calls:
                print(assistant_message)
                continue
            
            # Process all tool calls in the response
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                rationale = function_args.pop("rationale", "No rationale provided")
                
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
                elif function_name == "finish":
                    current_grid = grid_ops.get_grid()
                    current_image = self._grid_to_image(current_grid)
                    return current_grid, function_args["confidence"]
                
                # Add tool response to message history
                tool_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps({"status": "success"})
                }
                self.message_history.append(tool_response)
                self._save_message_history()
                messages.append(tool_response)
            
            # Save intermediate grid state after all tool calls
            current_grid = grid_ops.get_grid()
            state = {
                "step": step,
                "grid": current_grid,
                "tool": "multiple" if len(assistant_message.tool_calls) > 1 else assistant_message.tool_calls[0].function.name,
                "description": f"After {len(assistant_message.tool_calls)} operations",
                "rationale": rationale
            }
            self.intermediate_grids.append(state)
            self._save_intermediate_state(**state)
            step += 1
            
            # Add the current grid state to the message history
            current_image = self._grid_to_image(current_grid)
            grid_update = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Current output grid state:\n{json.dumps(current_grid)}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Is this what you expected?"
                    }
                ]
            }
            self.message_history.append(grid_update)
            self._save_message_history()
            messages.append(grid_update) 