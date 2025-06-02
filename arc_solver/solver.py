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
    """
    Manages the process of solving Abstraction and Reasoning Corpus (ARC) tasks
    using an OpenAI model and a defined set of grid manipulation tools.

    The solver initializes with an API key, prepares task data for the model,
    iteratively calls the model to get reasoning and tool usage, applies tools
    via GridOperations, and tracks conversation history, intermediate grid states,
    and token usage.
    """
    def __init__(self, api_key: str, results_dir: str = None):
        """
        Initializes the ARCSolver.

        Args:
            api_key (str): The OpenAI API key.
            results_dir (str, optional): The directory where results and intermediate
                                         states for tasks will be saved. If None,
                                         saving to disk is disabled for some parts.
                                         Defaults to None.
        """
        self.client = OpenAI(api_key=api_key) # OpenAI API client
        self.tools = self._define_tools()      # Load tool definitions from JSON

        # State variables per task, reset in solve_task
        self.message_history = []              # Stores the conversation with the model
        self.intermediate_grids = []           # Stores grid states after tool applications
        self.total_input_tokens = 0            # Tracks input tokens for the current task
        self.total_output_tokens = 0           # Tracks output tokens for the current task

        # Configuration
        self.results_dir = results_dir                # Base directory for saving results
        self.current_task_dir = None                  # Specific directory for the current task's results
        self.budget = 2000000                         # Token budget for solving a task (approximate)
        
    def get_message_history(self) -> List[Dict]:
        """
        Retrieves the message history for the current task, ensuring all messages
        are in a serializable dictionary format.

        Returns:
            List[Dict]: A list of message dictionaries.
        """
        return [self._message_to_dict(msg) for msg in self.message_history]
        
    def get_intermediate_grids(self) -> List[Dict]:
        """
        Retrieves the list of intermediate grid states recorded during the task solution process.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents
                        an intermediate state (step, grid, tool, description).
        """
        return self.intermediate_grids
        
    def get_token_counts(self) -> Dict[str, int]:
        """
        Retrieves the total input, output, and combined token counts for the current task.

        Returns:
            Dict[str, int]: A dictionary with keys "input_tokens", "output_tokens",
                            and "total_tokens".
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
        
    def _define_tools(self) -> List[Dict]:
        """
        Loads tool definitions from the `tools/tool_definitions.json` file.

        The path to the JSON file is constructed relative to this script's location.
        Includes error handling for file not found or malformed JSON.

        Returns:
            List[Dict]: A list of tool definitions. Returns an empty list if
                        loading fails.
        """
        tools_file_path = os.path.join(os.path.dirname(__file__), "tools", "tool_definitions.json")
        try:
            with open(tools_file_path, 'r') as f:
                tools = json.load(f)
            # Validate that the loaded data is a list, as expected for tool definitions.
            if not isinstance(tools, list):
                print(f"Error: tool_definitions.json does not contain a valid list. Found type: {type(tools)}")
                return []
            return tools
        except FileNotFoundError:
            print(f"Error: Tool definitions file not found at {tools_file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {tools_file_path}: {e}")
            return []
        except Exception as e: # Catch any other unexpected errors during loading.
            print(f"An unexpected error occurred while loading tool definitions: {e}")
            return []

    def _grid_to_image(self, grid: List[List[int]]) -> str:
        """
        Converts a 2D list representing an ARC grid into a base64-encoded PNG image.

        The image visualization is designed for clarity:
        - Each grid cell (tile) is rendered as a 14x14 pixel square.
        - Each tile is centered within a 16x16 pixel "patch" area, with a 1px border.
        - Coordinate labels (0-indexed) are displayed on the top and left edges.
        - The image is padded to standard sizes (144x144, 256x256, or 512x512)
          depending on the grid dimensions to maintain a consistent aspect ratio
          for the model, if these images were to be used as direct model inputs.
        - For grids that would result in an image larger than 512x512 (after
          accounting for labels and padding), the displayed grid is truncated to
          fit within the 512x512 image, showing the top-left portion.

        Args:
            grid (List[List[int]]): The ARC grid, a list of lists of integers.

        Returns:
            str: A base64-encoded string representing the PNG image.
        """
        if not grid or not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            # Handle empty or malformed grid gracefully
            placeholder_img = Image.new('RGB', (144, 144), (230, 230, 230)) # Light grey
            draw = ImageDraw.Draw(placeholder_img)
            try:
                # Attempt to load a simple font, fallback if not found
                font = ImageFont.truetype("arial.ttf", 10)
            except IOError:
                font = ImageFont.load_default()
            draw.text((10, 10), "Invalid/Empty Grid", fill=(0,0,0), font=font)
            buffered = BytesIO()
            placeholder_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        arr = np.array(grid, dtype=np.uint8)
        # Standard ARC color mapping
        colors = {
            0: (0, 0, 0),       # Black
            1: (0, 116, 217),   # Blue
            2: (255, 65, 54),    # Red
            3: (46, 204, 64),    # Green
            4: (255, 220, 0),   # Yellow
            5: (170, 170, 170),   # Grey
            6: (240, 18, 190),    # Magenta/Pink
            7: (255, 133, 27),   # Orange
            8: (127, 219, 255),   # Light Blue
            9: (135, 12, 37)     # Dark Red/Maroon
        }

        # Image rendering constants
        PATCH_SIZE = 16  # Total size of a "patch" (cell area + padding)
        TILE_SIZE = 14   # Actual colored square size for a cell
        BORDER = (PATCH_SIZE - TILE_SIZE) // 2 # Border around each tile within its patch
        LABEL_AREA_SIZE = 12 # Space for coordinate labels
        LABEL_PADDING = 8    # Padding around labels

        height, width = arr.shape

        # Calculate total required width and height for the grid content including labels
        content_width = LABEL_AREA_SIZE + LABEL_PADDING + width * PATCH_SIZE
        content_height = LABEL_AREA_SIZE + LABEL_PADDING + height * PATCH_SIZE
        
        # Determine the final image size based on content dimensions
        min_required_dimension = max(content_width, content_height)
        if min_required_dimension <= 144:
            final_size = 144
        elif min_required_dimension <= 256:
            final_size = 256
        else:
            final_size = 512 # Max standard size

        # Create the base image (white background)
        img = Image.new('RGB', (final_size, final_size), (255, 255, 255))
        pixels = img.load() # For direct pixel manipulation

        # Determine how many grid cells can actually be drawn if content exceeds final_size
        drawable_width = min(width, (final_size - LABEL_AREA_SIZE - LABEL_PADDING) // PATCH_SIZE)
        drawable_height = min(height, (final_size - LABEL_AREA_SIZE - LABEL_PADDING) // PATCH_SIZE)

        # Calculate offset to center the entire grid block (content) within the final image
        offset_x = max(0, (final_size - (LABEL_AREA_SIZE + LABEL_PADDING + drawable_width * PATCH_SIZE)) // 2)
        offset_y = max(0, (final_size - (LABEL_AREA_SIZE + LABEL_PADDING + drawable_height * PATCH_SIZE)) // 2)

        # Draw the grid tiles
        for r in range(drawable_height): # r for row index
            for c in range(drawable_width): # c for column index
                color_value = arr[r, c]
                rgb_color = colors.get(color_value, (0,0,0)) # Default to black if color not in map

                # Calculate top-left corner of the patch for this cell
                patch_origin_x = offset_x + LABEL_AREA_SIZE + LABEL_PADDING + c * PATCH_SIZE
                patch_origin_y = offset_y + LABEL_AREA_SIZE + LABEL_PADDING + r * PATCH_SIZE

                # Calculate top-left corner of the actual colored tile (centered in patch)
                tile_origin_x = patch_origin_x + BORDER
                tile_origin_y = patch_origin_y + BORDER

                # Draw the tile
                for i in range(TILE_SIZE):
                    for j in range(TILE_SIZE):
                        # Ensure drawing is within image bounds (should be, due to drawable_width/height)
                        if tile_origin_x + j < final_size and tile_origin_y + i < final_size:
                            pixels[tile_origin_x + j, tile_origin_y + i] = rgb_color
        
        # Prepare to draw text labels
        draw = ImageDraw.Draw(img)
        try:
            # Attempt to load a common font, fallback to default if not available
            font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default()

        # Draw X-axis labels (column numbers)
        for c in range(drawable_width):
            label_text = str(c)
            # Position label centered above the patch
            text_x = offset_x + LABEL_AREA_SIZE + LABEL_PADDING + c * PATCH_SIZE + (PATCH_SIZE // 2) - (font.getbbox(label_text)[2] // 2)
            text_y = offset_y + LABEL_PADDING // 2
            draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font)
        
        # Draw Y-axis labels (row numbers)
        for r in range(drawable_height):
            label_text = str(r)
            # Position label centered to the left of the patch
            text_x = offset_x + LABEL_PADDING // 2
            text_y = offset_y + LABEL_AREA_SIZE + LABEL_PADDING + r * PATCH_SIZE + (PATCH_SIZE // 2) - (font.getbbox(label_text)[3] // 2)
            draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font)

        # Save image to a byte buffer and encode as base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

        return base64.b64encode(buffered.getvalue()).decode()

    def _save_intermediate_state(self, step: int, grid: List[List[int]], tool: str, description: str):
        """
        Saves the current grid state, tool used, and description to disk if a
        results directory is configured for the current task.

        This method appends to a 'states.json' file and also saves individual
        JSON and PNG files for each step.

        Args:
            step (int): The current step number in the solution process.
            grid (List[List[int]]): The current state of the grid.
            tool (str): The name of the tool or action that led to this state.
            description (str): A description of the action taken.
        """
        if not self.current_task_dir: # Results saving disabled if no task directory
            return
            
        # Ensure the directory for intermediate states exists
        intermediate_dir = os.path.join(self.current_task_dir, "intermediate_states")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Prepare data for this state
        state_data = {
            "step": step,
            "grid": grid, # Grid is stored as list of lists
            "tool": tool,
            "description": description
        }
        
        # Append to the main 'states.json' list for the task
        states_file_path = os.path.join(intermediate_dir, "states.json")
        try:
            if os.path.exists(states_file_path):
                with open(states_file_path, "r") as f:
                    all_states_data = json.load(f)
            else:
                all_states_data = []
            all_states_data.append(state_data)
            with open(states_file_path, "w") as f:
                json.dump(all_states_data, f, indent=2)
        except Exception as e:
            print(f"Error saving to states.json: {e}")

        # Save as an individual JSON file for this step
        step_json_path = os.path.join(intermediate_dir, f"step_{step:03d}.json")
        try:
            with open(step_json_path, "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"Error saving step JSON {step_json_path}: {e}")
            
        # Save the grid as a PNG image for this step
        try:
            img_base64_data = self._grid_to_image(grid)
            img_bytes = base64.b64decode(img_base64_data)
            img_path = os.path.join(intermediate_dir, f"step_{step:03d}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print(f"Error saving step image {img_path}: {e}")
            
    def _message_to_dict(self, message) -> Dict:
        """
        Converts an OpenAI message object (or a dict) to a JSON-serializable dictionary.

        This handles messages from the OpenAI API which might be specific objects
        (like `Message` or `FunctionCall`) and ensures they can be logged as JSON.

        Args:
            message: The message object or dictionary.

        Returns:
            Dict: A JSON-serializable dictionary representation of the message.
        """
        if hasattr(message, 'model_dump'): # Standard for Pydantic-based models in newer OpenAI SDK
            return message.model_dump()
        elif isinstance(message, dict): # If it's already a dict
            return message
        else: # Fallback for other types, convert to string
            return str(message)
            
    def _save_message_history(self):
        """
        Saves the current message history to 'message_history.json' in the current
        task's result directory, if configured.
        """
        if not self.current_task_dir: # Results saving disabled
            return
            
        history_file_path = os.path.join(self.current_task_dir, "message_history.json")
        try:
            # Ensure all messages are converted to serializable dictionaries
            serializable_history = [self._message_to_dict(msg) for msg in self.message_history]
            with open(history_file_path, "w") as f:
                json.dump(serializable_history, f, indent=2)
        except Exception as e:
            print(f"Error saving message history: {e}")
        
    def solve_task(self, task_json: Dict, input_grid: List[List[int]], evolved_instructions: str, task_name: str) -> Tuple[Optional[List[List[int]]], int]:
        """
        Solves a given ARC task using an iterative, tool-based approach with an LLM.

        The process involves:
        1. Initializing state (message history, grid operations, token counts).
        2. Preparing system and user messages, including task examples and instructions.
        3. Entering a loop where the LLM is called:
            a. The LLM provides reasoning and may request function calls (tools).
            b. If tools are called, they are executed, and the grid is updated.
            c. The conversation history and grid state are updated and saved.
            d. The loop continues until the LLM calls the 'finish' tool or an error/budget limit occurs.

        Args:
            task_json (Dict): The ARC task data in JSON format, containing "train" and "test" examples.
            input_grid (List[List[int]]): The specific input grid for the test case to be solved.
            evolved_instructions (str): The main prompt/instructions for the LLM, potentially evolved.
            task_name (str): A unique name for the task, used for creating results directories.

        Returns:
            Tuple[Optional[List[List[int]]], int]:
                - The final predicted output grid (as a list of lists of integers), or None if an error caused premature termination.
                - The confidence score (0-10) provided by the LLM when calling 'finish', or 0 if unfinished/error.
        """
        # --- Initialization and Setup ---
        self.message_history = []  # Reset for the new task
        self.intermediate_grids = [] # Reset for the new task
        self.total_input_tokens = 0  # Reset token counters
        self.total_output_tokens = 0

        self.current_task_dir = None # Reset current task directory
        if self.results_dir: # If a base results directory is provided
            self.current_task_dir = os.path.join(self.results_dir, task_name)
            os.makedirs(self.current_task_dir, exist_ok=True) # Create task-specific subdir
        
        grid_ops = GridOperations(input_grid) # Initialize grid operations with the current task's input
        
        # Save the initial state of the grid (step 0)
        initial_state_data = {
            "step": 0,
            "grid": grid_ops.get_grid(),
            "tool": "initial",
            "description": "Initial grid state"
        }
        self.intermediate_grids.append(initial_state_data)
        self._save_intermediate_state(**initial_state_data) # Save to disk if current_task_dir is set
        
        # --- System Message Preparation ---
        # This message defines the LLM's role, provides general instructions, and includes the (potentially evolved) task-solving strategy.
        system_message_content = (
            "You are an ARC grid-puzzle solver. You will be given task demonstrations, from which you can infer "
            "rules and patterns that define the task. You will then be given an input grid and tools to generate "
            "the solution/output grid.\n"
            f"{evolved_instructions}"  # Specific strategy/prompt for solving
            "\n\nFirst, return your reasoning about what the underlying rules and possible solution should be. "
            "Next, execute the tool calls to generate the output grid (never output the grid directly.) "
            "Once each tool call returns, the user will send the updated representation of the output grid. "
            "Continue with the tool calls until you are confident in your solution, then use the finish tool "
            "with a confidence score. Never send a message without a tool call!"
        )
        system_message = {"role": "developer", "content": system_message_content}
        self.message_history.append(system_message)
        self._save_message_history() # Save history if current_task_dir is set
        
        # --- User Message Preparation (Task Context) ---
        # This message provides all the task details: training examples and the specific test input.
        user_message_parts = []
        
        # Add training examples (input-output pairs)
        for i, demo_pair in enumerate(task_json.get("train", []), 1):
            user_message_parts.extend([
                {"type": "input_text", "text": f"Training input {i}:\n{json.dumps(demo_pair['input'])}"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{self._grid_to_image(demo_pair['input'])}"},
                {"type": "input_text", "text": f"Training output {i}:\n{json.dumps(demo_pair['output'])}"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{self._grid_to_image(demo_pair['output'])}"}
            ])
        
        # Add the current test input grid for which a solution is sought
        user_message_parts.extend([
            {"type": "input_text", "text": f"\nTask input grid:\n{json.dumps(input_grid)}"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{self._grid_to_image(input_grid)}"}
        ])
        
        user_message = {"role": "user", "content": user_message_parts}
        self.message_history.append(user_message)
        self._save_message_history()
        
        # Initialize the list of messages to be sent to the LLM
        messages_for_api = [system_message, user_message]
        
        # --- Main Solving Loop ---
        current_step = 1
        max_api_call_retries = 3 # Max retries for a single API call

        while True: # Loop continues until 'finish' tool is called or an error/budget limit
            # --- API Call to LLM ---
            api_retry_count = 0
            response = None # Initialize response to None
            while api_retry_count < max_api_call_retries:
                try:
                    # Call the OpenAI API (using the "responses" endpoint which supports tools)
                    response = self.client.responses.create(
                        model="o4-mini", # Specify the model
                        input=messages_for_api, # Current conversation history
                        tools=self.tools,       # Available tools
                        tool_choice="auto",     # Let the model decide if/which tool to use
                        reasoning={"effort": "medium", "summary": "auto"}, # Request reasoning
                        truncation="auto"
                        # store=False, # Optional: affects storage on OpenAI side
                        # include=["reasoning.encrypted_content"] # Optional: for encrypted reasoning
                    )
                    
                    # Track token usage if available in the response
                    if hasattr(response, 'usage') and response.usage:
                        print(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}, Total: {response.usage.total_tokens}")
                        # Optional: more detailed token info if needed for debugging
                        # if hasattr(response.usage, 'input_tokens_details'):
                        #     print(f"Input tokens details - Cached: {response.usage.input_tokens_details.cached_tokens}")
                        # if hasattr(response.usage, 'output_tokens_details'):
                        #     print(f"Output tokens details - Reasoning: {response.usage.output_tokens_details.reasoning_tokens}")
                        
                        self.total_input_tokens += response.usage.input_tokens
                        self.total_output_tokens += response.usage.output_tokens
                    
                    break # API call successful, exit retry loop
                    
                except Exception as e: # Catch API errors (network issues, rate limits, etc.)
                    api_retry_count += 1
                    print(f"Error in API call (attempt {api_retry_count}/{max_api_call_retries}): {str(e)}")
                    
                    if api_retry_count == max_api_call_retries:
                        print("Max API retries reached. Terminating task.")
                        return None, 0  # Return None grid and 0 confidence
                    
                    time.sleep(2 ** api_retry_count) # Exponential backoff before retrying

            if response is None: # Should not happen if retry logic is correct, but as a safeguard
                print("API response was unexpectedly None. Terminating task.")
                return None, 0

            # Append the model's output (which can include reasoning, function calls) to our context for the next API call
            messages_for_api.extend(response.output) # Note: response.output is a list of message parts

            # --- Process Model's Response (Reasoning & Function Calls) ---
            tool_called_in_this_turn = False
            for item in response.output: # Iterate through parts of the model's response
                if item.type == "reasoning":
                    # Log reasoning if present
                    summary_texts = [summary.text for summary in item.summary if summary.text]
                    if summary_texts:
                        self.message_history.append({
                            "role": "assistant", # Model's reasoning
                            "content": "Reasoning:\n" + "\n".join(summary_texts)
                        })
                        self._save_message_history()

                elif item.type == "function_call":
                    tool_called_in_this_turn = True
                    # Prepare the function call details for logging and execution
                    function_call_details = {
                        "type": "function_call", # Standard field for API
                        "call_id": item.call_id, # ID of the call
                        "name": item.name,       # Name of the function to call
                        "arguments": item.arguments # Arguments as a JSON string
                    }
                    self.message_history.append(function_call_details) # Log the raw function call
                    self._save_message_history()
                    
                    function_name = item.name
                    try:
                        function_args = json.loads(item.arguments) # Parse arguments from JSON string
                    except json.JSONDecodeError as e:
                        print(f"Error decoding function arguments for {function_name}: {e}. Arguments: {item.arguments}")
                        # Prepare an error response for the model
                        tool_response_content = f"Error: Could not parse arguments for function {function_name}. Invalid JSON: {e}"
                        tool_response_message = {"type": "function_call_output", "call_id": item.call_id, "output": tool_response_content}
                        messages_for_api.append(tool_response_message)
                        self.message_history.append(tool_response_message)
                        self._save_message_history()
                        continue # Skip to next item in response.output

                    # --- Tool Dispatching and Execution ---
                    tool_execution_status = "success" # Assume success initially
                    tool_output_message = ""          # For error messages from tools

                    if function_name == "fill_tiles":
                        grid_ops.fill_tiles(function_args.get("positions", []))
                    elif function_name == "copy_grid":
                        grid_ops.copy_grid()
                    elif function_name == "fill_pattern":
                        grid_ops.fill_pattern(
                            function_args.get("start_x"), function_args.get("start_y"),
                            function_args.get("direction"), function_args.get("interval"),
                            function_args.get("color")
                        )
                    elif function_name == "fill_rectangle":
                        grid_ops.fill_rectangle(
                            function_args.get("x1"), function_args.get("y1"),
                            function_args.get("x2"), function_args.get("y2"),
                            function_args.get("color")
                        )
                    elif function_name == "translate":
                        grid_ops.translate(function_args.get("dx"), function_args.get("dy"))
                    elif function_name == "resize_grid":
                        grid_ops.resize_grid(function_args.get("width"), function_args.get("height"))
                    elif function_name == "copy_selection":
                        grid_ops.copy_selection(
                            function_args.get("start_x"), function_args.get("start_y"),
                            function_args.get("end_x"), function_args.get("end_y"),
                            function_args.get("paste_origins", [])
                        )
                    elif function_name == "execute_python":
                        py_result = grid_ops.execute_python_code(function_args.get("code", ""))
                        if not py_result["success"]:
                            tool_execution_status = "error"
                            tool_output_message = py_result["message"]
                    elif function_name == "finish":
                        # 'finish' tool indicates completion
                        final_grid = grid_ops.get_grid()
                        confidence = function_args.get("confidence", 0) # Default to 0 confidence if not provided
                        # Log the finish call details
                        self.message_history.append({
                            "role": "assistant",
                            "content": f"Called finish tool with confidence {confidence}. Rationale: {function_args.get('rationale', 'N/A')}"
                        })
                        self._save_message_history()
                        return final_grid, confidence
                    else:
                        # Unknown function name
                        tool_execution_status = "error"
                        tool_output_message = f"Error: Unknown tool '{function_name}' called."
                    
                    # --- Prepare and Log Tool Response ---
                    # The API expects a "function_call_output" message after a "function_call"
                    if tool_execution_status == "success":
                        tool_response_content = json.dumps({"status": "success"}) # Simple success message
                    else:
                        tool_response_content = tool_output_message # Error message from tool execution

                    tool_response_message = {
                        "type": "function_call_output",
                        "call_id": item.call_id, # Must match the call_id of the function_call
                        "output": tool_response_content
                    }
                    messages_for_api.append(tool_response_message) # Add to context for next API call
                    self.message_history.append(tool_response_message) # Log it
                    self._save_message_history()

                else: # Handle other message types if any (e.g., "message" type though less common with tool use)
                    self.message_history.append(self._message_to_dict(item)) # Ensure serializable
                    self._save_message_history()

            # If no tool was called in this turn (e.g., model just returned reasoning or text),
            # it might indicate an issue or a need to reprompt. For now, we print a message.
            if not tool_called_in_this_turn:
                print("Warning: Model response did not include a tool call in this turn.")
                # Potentially, one could add logic here to reprompt or terminate if this happens too often.
                # For now, the loop continues, but the grid state won't change.
                # If the model is stuck, it might keep returning non-tool messages.
                # Consider adding a counter for consecutive non-tool turns.
                # However, the system prompt explicitly says "Never send a message without a tool call!"
                # so this state should ideally be rare.
                pass # Continue to next iteration or grid update phase

            # --- Save Intermediate Grid State and Prepare for Next Iteration ---
            current_grid_state = grid_ops.get_grid()
            # Collect names of tools called in this turn for description
            function_calls_this_turn = [fc_item for fc_item in response.output if fc_item.type == "function_call"]
            tool_names_this_turn = [fc.name for fc in function_calls_this_turn]
            
            current_state_data = {
                "step": current_step,
                "grid": current_grid_state,
                "tool": "multiple" if len(tool_names_this_turn) > 1 else tool_names_this_turn[0] if tool_names_this_turn else "none",
                "description": ", ".join(tool_names_this_turn) if tool_names_this_turn else "No tool executed (e.g. only reasoning)"
            }
            self.intermediate_grids.append(current_state_data)
            self._save_intermediate_state(**current_state_data)
            current_step += 1

            # --- Clean up previous 'Current output grid state' messages for brevity in context ---
            # This helps manage context length by replacing older, detailed grid states with a placeholder.
            for i, msg_content in enumerate(messages_for_api):
                if (isinstance(msg_content, dict) and
                    msg_content.get("role") == "user" and
                    isinstance(msg_content.get("content"), list) and
                    len(msg_content["content"]) > 0 and
                    isinstance(msg_content["content"][0], dict) and
                    msg_content["content"][0].get("type") == "input_text" and
                    msg_content["content"][0].get("text", "").startswith("Current output grid state:")):
                    messages_for_api[i] = { # Replace with a summarized version
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Current output grid state: [grid representation removed for brevity]"}]
                    }
            
            # --- Add Updated Grid State for Next Model Turn ---
            # The user (solver system) provides the new grid state back to the LLM.
            current_grid_image_b64 = self._grid_to_image(current_grid_state)
            grid_update_message = {
                "role": "user", # Simulating user providing the updated grid
                "content": [
                    {"type": "input_text", "text": f"Current output grid state:\n{json.dumps(current_grid_state)}"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{current_grid_image_b64}"},
                    {"type": "input_text", "text": "Is this what you expected? (don't reply, just think about it as you solve the task)"}
                ]
            }
            messages_for_api.append(grid_update_message)
            self.message_history.append(grid_update_message) # Log this update
            self._save_message_history()

            # --- Check Solving Budget ---
            # If token usage exceeds budget, terminate to prevent runaway costs.
            if self.total_input_tokens + self.total_output_tokens > self.budget: # Check combined tokens
                print(f"Solving budget of {self.budget} tokens exceeded. Current total: {self.total_input_tokens + self.total_output_tokens}. Outputting current grid.")
                return grid_ops.get_grid(), 0 # Return current grid with 0 confidence