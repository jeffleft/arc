import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.ai.generativelanguage as glm
from grid_ops import GridOperations
import os
import time

# Helper function to convert schema
def convert_schema(schema_dict):
    if not isinstance(schema_dict, dict):
        return schema_dict

    # Create a copy to avoid modifying the original
    new_schema = schema_dict.copy()

    if "type" in new_schema:
        type_str = new_schema.pop("type").upper()
        # Ensure TYPE_UNSPECIFIED is not used if not explicitly set
        if type_str == "OBJECT":
            new_schema["type_"] = glm.Type.OBJECT
        elif type_str == "ARRAY":
            new_schema["type_"] = glm.Type.ARRAY
        elif type_str == "STRING":
            new_schema["type_"] = glm.Type.STRING
        elif type_str == "NUMBER": # OpenAPI uses "number" for float and double
            new_schema["type_"] = glm.Type.NUMBER
        elif type_str == "INTEGER":
            new_schema["type_"] = glm.Type.INTEGER
        elif type_str == "BOOLEAN":
            new_schema["type_"] = glm.Type.BOOLEAN
        # Add other type mappings if necessary

    if "properties" in new_schema:
        new_schema["properties"] = {
            k: convert_schema(v) for k, v in new_schema["properties"].items()
        }

    if "items" in new_schema:
        new_schema["items"] = convert_schema(new_schema["items"])

    return new_schema

class ARCSolver:
    def __init__(self, api_key: str, results_dir: str = None):
        """Initialize the ARC solver with Gemini API key."""
        genai.configure(api_key=api_key)

        # Convert tool definitions
        raw_tools = self._get_raw_tool_definitions()
        converted_tools = []
        for tool_def in raw_tools:
            converted_parameters = None
            if "parameters" in tool_def and tool_def["parameters"]:
                 # Pass the whole parameters dict to convert_schema
                converted_parameters_dict = convert_schema(tool_def["parameters"])
                converted_parameters = glm.Schema(**converted_parameters_dict)

            converted_tools.append(
                glm.Tool(function_declarations=[
                    glm.FunctionDeclaration(
                        name=tool_def["name"],
                        description=tool_def["description"],
                        parameters=converted_parameters
                    )
                ])
            )

        self.client = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            tools=converted_tools,
            safety_settings={ # Add safety settings to avoid blocking
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        self.message_history = []
        self.intermediate_grids = []
        self.results_dir = results_dir
        self.current_task_dir = None
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
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "candidates_token_count": 0, # Placeholder for Gemini specific counts
            "prompt_token_count": 0 # Placeholder
        }

    def _get_raw_tool_definitions(self) -> List[Dict]:
        """Get the raw tool definitions before conversion."""
        # This method now just returns the list of dictionaries.
        # The conversion to glm.Tool and glm.FunctionDeclaration happens in __init__.
        return [
            {
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
            },
            {
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
                                "type": "array", # This nested array needs careful handling
                                "description": "(x, y) tuples",
                                "items": { # Items of the inner array
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
                "name": "resize_grid",
                "description": "Resize the output grid to MxN dimensions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "width": {
                            "type": "integer",
                            "description": "The new width of the grid (integer, min: 1, max: 100)"
                        },
                        "height": {
                            "type": "integer",
                            "description": "The new height of the grid (integer, min: 1, max: 100)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["width", "height", "rationale"]
                }
            },
            {
                "name": "fill_tiles",
                "description": "Fill specific tiles with given colors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positions": {
                            "type": "array",
                            "items": { # Items of the 'positions' array (which are objects)
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                    "color": {"type": "integer"}
                                },
                                "required": ["x", "y", "color"]
                            }
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this operation is being performed"
                        }
                    },
                    "required": ["positions", "rationale"]
                }
            },
            {
                "name": "finish",
                "description": "Indicate that the solution is complete",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confidence": {
                            "type": "integer",
                            "description": "Confidence score from 0-10 in the solution's correctness (integer, min: 0, max: 10)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why the solution is complete and why the confidence score was chosen"
                        }
                    },
                    "required": ["confidence", "rationale"]
                }
            },
            {
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
                    "required": ["code", "rationale"]
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
            # Recursively process parts of the message if it's a dictionary
            # This is important for messages that have 'parts' which might contain images
            processed_message = {}
            for key, value in message.items():
                if key == "parts" and isinstance(value, list):
                    processed_message[key] = [self._message_to_dict(part) for part in value]
                elif isinstance(value, Image.Image): # Handle PIL Image objects in parts
                    buffered = BytesIO()
                    value.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    processed_message[key] = {"type": "image_base64", "data": img_str, "format": value.format}
                else:
                    processed_message[key] = value # Keep other parts as is, assuming they are serializable
            return processed_message
        elif isinstance(message, Image.Image): # Handle if a message part itself is an Image
            buffered = BytesIO()
            message.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            # Return a dictionary structure that indicates this was an image
            return {"type": "image_base64", "data": img_str, "format": message.format}
        elif hasattr(message, 'parts') and isinstance(message.parts, list): # For glm.Content objects
            # Create a dict representation, and process its parts
            content_dict = {"role": message.role, "parts": [self._message_to_dict(part) for part in message.parts]}
            return content_dict
        elif hasattr(message, 'text') and isinstance(message.text, str): # For glm.Part with text
            return {"type": "text", "text": message.text}
        # Add more specific handlers if other non-serializable types are encountered in messages
        else:
            # Fallback for other types
            try:
                # Attempt to convert to string as a last resort
                return str(message)
            except Exception:
                # If str() fails, provide a placeholder to prevent crashing serialization
                return f"Object of type {message.__class__.__name__} is not JSON serializable (and failed str conversion)"

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

        # reset token counts
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
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
            "role": "system",  # Gemini uses "system" role for system messages
            "parts": [  # Gemini uses "parts" for content
                ("You are an ARC grid-puzzle solver. You will be given task demonstrations, from which you can infer "
                 "rules and patterns that define the task. You will then be given an input grid and tools to generate "
                 "the solution/output grid.\n"
                 f"{evolved_instructions}"
                 "\n\nFirst, return your reasoning about what the underlying rules and possible solution should be. "
                 "Next, execute the tool calls to generate the output grid (never output the grid directly.) "
                 "Once each tool call returns, the user will send the updated representation of the output grid. "
                 "Continue with the tool calls until you are confident in your solution, then use the finish tool "
                 "with a confidence score. Never send a message without a tool call!")
            ]
        }
        
        # Add system message to log (system_message is already a dict)
        self.message_history.append(system_message)
        self._save_message_history()
        
        # Prepare the user message with context
        # user_message_parts will contain actual Image objects
        user_message_parts = []
        
        # Add training examples sequentially
        for i, demo in enumerate(task_json.get("train", []), 1):
            # Add input
            user_message_parts.append(f"Training input {i}:\n{json.dumps(demo['input'])}")
            user_message_parts.append(Image.open(BytesIO(base64.b64decode(self._grid_to_image(demo['input'])))))
            
            # Add output
            user_message_parts.append(f"Training output {i}:\n{json.dumps(demo['output'])}")
            user_message_parts.append(Image.open(BytesIO(base64.b64decode(self._grid_to_image(demo['output'])))))
        
        # Add the current input grid
        user_message_parts.append(f"\nTask input grid:\n{json.dumps(input_grid)}")
        user_message_parts.append(Image.open(BytesIO(base64.b64decode(self._grid_to_image(input_grid)))))
        
        user_message = {
            "role": "user",
            "parts": user_message_parts
        }

        # Add user message to log. user_message is a dict with 'role' and 'parts'.
        # The 'parts' can contain Image objects. _message_to_dict will handle them.
        self.message_history.append(user_message)
        self._save_message_history()
        
        # Initialize chat history for the Gemini model.
        # This needs to be a list of glm.Content objects.
        # System message is already a dict, user_message parts include Image objects.
        # We need to convert these to glm.Content before starting the chat.

        chat_history_for_model = []
        # Convert system message dict to glm.Content
        if system_message["role"] == "system": # Gemini SDK might prefer "system" for system instructions.
                                              # Or it might be part of the model config, not chat history.
                                              # For now, let's assume it's part of history if needed.
                                              # The SDK usually takes system_instruction separately.
                                              # Let's assume our current structure of just passing it as a message is fine.
            chat_history_for_model.append(glm.Content(role="user", parts=[glm.Part(text=system_message["parts"][0])])) # Simplified for now
        
        # Convert initial user message (with images) to glm.Content
        glm_user_parts = []
        for part in user_message["parts"]:
            if isinstance(part, Image.Image):
                # Convert PIL Image to glm.Part with inline_data (Blob)
                buffered = BytesIO()
                part.save(buffered, format="PNG") # Assuming PNG, adjust if other formats are used
                img_bytes = buffered.getvalue()
                blob = glm.Blob(mime_type="image/png", data=img_bytes)
                glm_user_parts.append(glm.Part(inline_data=blob))
            else: # Assuming text parts
                glm_user_parts.append(glm.Part(text=str(part)))
        chat_history_for_model.append(glm.Content(role="user", parts=glm_user_parts))

        # `messages_for_log` list will store the dict representations for JSON logging
        messages_for_log = [self._message_to_dict(msg) for msg in chat_history_for_model]

        # Loop until finish tool is called
        step = 1
        self.chat_session = None # Initialize chat session

        while True:
            # Get completion from Gemini API
            max_retries = 3
            retry_count = 0
            response = None # Ensure response is defined

            while retry_count < max_retries:
                try:
                    if self.chat_session is None:
                        # Filter out any non-Content objects from chat_history_for_model before starting chat
                        valid_chat_history = [msg for msg in chat_history_for_model if isinstance(msg, glm.Content)]
                        self.chat_session = self.client.start_chat(history=valid_chat_history)

                    # The content for send_message should be the parts of the last user message.
                    # The last message in chat_history_for_model is what we want to send.
                    last_message_content = chat_history_for_model[-1].parts
                    response = self.chat_session.send_message(last_message_content)
                    
                    # Token counting for Gemini
                    if hasattr(response, 'usage_metadata'):
                        if hasattr(response.usage_metadata, 'prompt_token_count'):
                            self.total_input_tokens += response.usage_metadata.prompt_token_count
                            print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
                        if hasattr(response.usage_metadata, 'candidates_token_count'):
                             self.total_output_tokens += response.usage_metadata.candidates_token_count
                             print(f"Candidates tokens: {response.usage_metadata.candidates_token_count}")
                        if hasattr(response.usage_metadata, 'total_token_count'):
                            print(f"Total tokens: {response.usage_metadata.total_token_count}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error in API call (attempt {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count == max_retries:
                        print("Max retries reached. Skipping this task.")
                        self.chat_session = None
                        return None, 0
                    
                    time.sleep(2 ** retry_count)

            if response is None: # Should not happen if retry logic is correct, but as a safeguard
                print("Failed to get response from API after retries.")
                self.chat_session = None
                return None, 0

            # Add model's response to chat_history_for_model (for next turn) and messages_for_log (for JSON)
            model_response_content = glm.Content(role="model", parts=response.parts)
            chat_history_for_model.append(model_response_content)
            messages_for_log.append(self._message_to_dict(model_response_content))
            self.message_history.append(self._message_to_dict(model_response_content)) # Update main history for saving
            self._save_message_history()

            # Process function calls
            for part in response.parts:
                if part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = {key: value for key, value in function_call.args.items()}

                    # Log the function call
                    self.message_history.append({
                        "role": "assistant", # Or "model" if Gemini distinguishes
                        "content": f"Function call: {function_name}({json.dumps(function_args)})"
                    })
                    self._save_message_history()
                    
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
                            tool_response_part = {"function_response": {
                                "name": function_name,
                                "response": {"content": result["message"]}
                            }}
                            messages.append({"role": "user", "parts": [tool_response_part]}) # Gemini expects tool response from user role
                            self.message_history.append({"role": "user", "parts": [tool_response_part]})
                            self._save_message_history()
                            continue
                    elif function_name == "finish":
                        current_grid = grid_ops.get_grid()
                        self.chat_session = None # Reset chat session for next task
                        return current_grid, function_args["confidence"]
                    
                    # Add tool response to messages for Gemini
                    function_response_content = glm.Content(
                        role="user", # Gemini expects function responses from 'user' role in the chat history
                        parts=[glm.Part(
                            function_response=glm.FunctionResponse(
                                name=function_name,
                                response={"content": json.dumps({"status": "success"})}
                            )
                        )]
                    )
                    messages.append(self._message_to_dict(function_response_content)) # For JSON log
                    self.message_history.append(self._message_to_dict(function_response_content))
                    self._save_message_history()
                    # Note: The actual `chat.history` for the Gemini SDK is updated internally by `send_message`.
                    # We are maintaining `messages` primarily for our own logging and potentially for restarting chats.

            # Check if response does not have a tool call
            has_function_call = any(part.function_call for part in response.parts) if response.parts else False
            if not has_function_call:
                print("Response does not have a tool call or parts are empty!")
                # If there's text output from the model without a function call, log it
                if response.text:
                    self.message_history.append({
                        "role": "model",
                        "content": "Reasoning (no function call):\n" + response.text
                    })
                    self._save_message_history()
                # If no function call and no text, it might be an issue or an empty response.
                # Depending on desired behavior, might need to retry or handle as error.
                # For now, if it was just reasoning, we continue to allow the user to send the next grid state.
                # If the model is stuck and not calling 'finish', this loop could go on.
                # Add a safety break or more sophisticated check if needed.
                if not response.text: # If truly empty response
                     print("Empty response from model and no function call.")
                     # Decide if this is an error or if we should just wait for user grid update
                     # For now, let's assume it's waiting for the next grid state if no text.

            # Save intermediate grid state only if there were function calls that modified the grid
            current_grid = grid_ops.get_grid() # Get current grid regardless
            if has_function_call:
                function_calls_in_response = [part.function_call for part in response.parts if part.function_call]
            state = {
                "step": step,
                "grid": current_grid,
                "tool": "multiple" if len(function_calls_in_response) > 1 else function_calls_in_response[0].name if function_calls_in_response else "none",
                "description": ", ".join([fc.name for fc in function_calls_in_response])
            }
            self.intermediate_grids.append(state)
            self._save_intermediate_state(**state)
            step += 1
            
            # Add the current grid state to the message history for the next turn with Gemini
            current_image_bytes = base64.b64decode(self._grid_to_image(current_grid))
            grid_update_parts = [
                f"Current output grid state:\n{json.dumps(current_grid)}",
                Image.open(BytesIO(current_image_bytes)),
                "Is this what you expected? (don't reply, just think about it as you solve the task)"
            ]

            # This becomes the new user message content for the next turn.
            # It will be added to chat_history_for_model before the next send_message call.

            user_update_glm_parts = []
            for part_content in grid_update_parts:
                if isinstance(part_content, Image.Image):
                    buffered = BytesIO()
                    part_content.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue()
                    blob = glm.Blob(mime_type="image/png", data=img_bytes)
                    user_update_glm_parts.append(glm.Part(inline_data=blob))
                else:
                    user_update_glm_parts.append(glm.Part(text=str(part_content)))

            user_update_content = glm.Content(role="user", parts=user_update_glm_parts)
            chat_history_for_model.append(user_update_content) # Add to history for next model call
            messages_for_log.append(self._message_to_dict(user_update_content)) # For JSON log
            self.message_history.append(self._message_to_dict(user_update_content)) # Update main history
            self._save_message_history()

            # check solving budget
            if (self.total_input_tokens + self.total_output_tokens) > self.budget:
                print("Solving budget exceeded. Outputting current grid.")
                self.chat_session = None # Reset chat session
                return grid_ops.get_grid(), 0