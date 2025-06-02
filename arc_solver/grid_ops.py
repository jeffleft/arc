from typing import List, Dict, Tuple, Union
import numpy as np

class GridOperations:
    """
    A class to perform various operations on a 2D grid represented by a NumPy array.
    The grid typically contains integer values representing colors or states.
    """
    def __init__(self, input_grid: List[List[int]]):
        """
        Initializes the GridOperations instance.

        The working grid is initialized with the input_grid. The input_grid itself
        is also stored for reference, particularly for operations like 'copy_grid'.

        Args:
            input_grid (List[List[int]]): The initial grid to operate on.
        """
        self.grid = np.array(input_grid, dtype=int)  # Main working grid
        self.input_grid = np.array(input_grid, dtype=int)  # Store a copy of the original input grid
        self.height, self.width = self.grid.shape
        
    def fill_tiles(self, positions: List[Dict[str, int]]) -> None:
        """
        Fills specific tiles on the grid with given colors.

        Each position in the list specifies the 'x' and 'y' coordinates and the 'color'
        to fill at that tile. Coordinates are 0-indexed.
        Assumes 'x', 'y', and 'color' are integers as per tool schema.

        Args:
            positions (List[Dict[str, int]]): A list of dictionaries, where each
                dictionary must contain 'x', 'y', and 'color' keys.
                Example: [{"x": 0, "y": 0, "color": 5}, {"x": 1, "y": 2, "color": 3}]
        """
        if not isinstance(positions, list):
            # Or raise an error, or log. For now, do nothing if format is wrong.
            return

        for pos in positions:
            if not isinstance(pos, dict):
                continue # Skip malformed position entries

            try:
                # Ensure keys exist and attempt conversion if types are not strictly enforced by caller
                x = int(pos.get("x", -1))
                y = int(pos.get("y", -1))
                color = int(pos.get("color", 0)) # Default color 0 if not specified
            except (ValueError, TypeError):
                # Skip if conversion fails for a position
                continue

            # Check if the coordinates are within the grid boundaries
            if 0 <= y < self.height and 0 <= x < self.width:
                self.grid[y, x] = color
            # else: an invalid coordinate is ignored
                
    def copy_grid(self) -> None:
        """
        Resets the working grid to be a copy of the original input grid.
        """
        self.grid = self.input_grid.copy()
        # Update height and width in case the input_grid had different dimensions (though typically not)
        self.height, self.width = self.grid.shape
        
    def fill_pattern(self, start_x: int, start_y: int, direction: str, interval: int, color: int) -> None:
        """
        Fills tiles in a repeating pattern (fixed interval) starting from a given
        point, either horizontally or vertically.

        Args:
            start_x (int): The starting x-coordinate for the pattern.
            start_y (int): The starting y-coordinate for the pattern.
            direction (str): The direction of the pattern, either "horizontal" or "vertical".
            interval (int): The spacing between filled tiles. An interval of 1 means
                            every tile in the direction is filled. An interval of 2
                            means every other tile is filled, etc. Must be positive.
            color (int): The color to fill the tiles with.
        """
        if interval <= 0:
            # Invalid interval, do nothing or log/raise an error.
            # Current behavior: silently returns.
            print(f"Warning: fill_pattern called with invalid interval: {interval}")
            return

        # Make direction check case-insensitive
        normalized_direction = direction.lower() if isinstance(direction, str) else ""

        if normalized_direction == "horizontal":
            # Iterate from start_y, filling tiles horizontally with the given interval
            if 0 <= start_y < self.height: # Ensure start_y is within bounds
                for x in range(start_x, self.width, interval):
                    if 0 <= x < self.width: # Ensure current x is within bounds
                        self.grid[start_y, x] = color
        elif normalized_direction == "vertical":
            # Iterate from start_x, filling tiles vertically with the given interval
            if 0 <= start_x < self.width: # Ensure start_x is within bounds
                for y in range(start_y, self.height, interval):
                    if 0 <= y < self.height: # Ensure current y is within bounds
                        self.grid[y, start_x] = color
        else:
            # Invalid direction string, do nothing or log.
            print(f"Warning: fill_pattern called with invalid direction: {direction}")
            return # Silently return for invalid direction
                    
    def fill_rectangle(self, x1: int, y1: int, x2: int, y2: int, color: int) -> None:
        """
        Fills a rectangular area on the grid with a specified color.

        The coordinates (x1, y1) and (x2, y2) define any two opposite corners
        of the rectangle. The rectangle includes its boundaries.

        Args:
            x1 (int): The x-coordinate of the first corner.
            y1 (int): The y-coordinate of the first corner.
            x2 (int): The x-coordinate of the second corner.
            y2 (int): The y-coordinate of the second corner.
            color (int): The color to fill the rectangle with.
        """
        # Determine the actual top-left and bottom-right corners
        rect_x1, rect_x2 = min(x1, x2), max(x1, x2)
        rect_y1, rect_y2 = min(y1, y2), max(y1, y2)
        
        # Ensure coordinates are within the grid boundaries by clipping
        # This prevents errors if the specified rectangle is partially or fully outside the grid.
        final_x1 = max(0, min(rect_x1, self.width - 1))
        final_x2 = max(0, min(rect_x2, self.width - 1))
        final_y1 = max(0, min(rect_y1, self.height - 1))
        final_y2 = max(0, min(rect_y2, self.height - 1))
        
        # Fill the rectangle if the clipped coordinates form a valid area
        if final_x1 <= final_x2 and final_y1 <= final_y2:
            self.grid[final_y1 : final_y2 + 1, final_x1 : final_x2 + 1] = color
        
    def translate(self, dx: int, dy: int) -> None:
        """
        Translates (shifts) the entire grid by a given offset (dx, dy).
        Areas of the grid that are shifted out of bounds are lost.
        Newly uncovered areas are filled with zeros (or the default background color).

        Args:
            dx (int): The amount to shift the grid along the x-axis (horizontal).
                      Positive values shift right, negative values shift left.
            dy (int): The amount to shift the grid along the y-axis (vertical).
                      Positive values shift down, negative values shift up.
        """
        # Create a new grid of the same shape and type, filled with a default value (e.g., 0)
        new_grid = np.zeros_like(self.grid)
        
        # Determine the slices for the source (old grid) and destination (new grid)
        # These slices define the overlapping region that will be copied.

        # Destination slice (where to place the old data in the new grid)
        dest_y_start = max(0, dy)
        dest_y_end = min(self.height, self.height + dy)
        dest_x_start = max(0, dx)
        dest_x_end = min(self.width, self.width + dx)
        
        # Source slice (what part of the old grid to copy)
        src_y_start = max(0, -dy)
        src_y_end = min(self.height, self.height - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(self.width, self.width - dx)
        
        # Perform the copy only if there's a valid overlapping region
        if (dest_y_start < dest_y_end and dest_x_start < dest_x_end and
            src_y_start < src_y_end and src_x_start < src_x_end):
            new_grid[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
                self.grid[src_y_start:src_y_end, src_x_start:src_x_end]
        
        self.grid = new_grid
        # self.height and self.width remain unchanged as the grid size itself doesn't change.
        
    def resize_grid(self, new_width: int, new_height: int) -> None:
        """
        Resizes the grid to new dimensions (new_width, new_height).

        If the new dimensions are smaller, the grid is cropped from the
        bottom-right. If larger, new areas are filled with zeros (or the
        default background color).

        Args:
            new_width (int): The new width of the grid. Must be positive.
            new_height (int): The new height of the grid. Must be positive.
        """
        if new_width <= 0 or new_height <= 0:
            # Invalid dimensions. The user/LLM should provide positive dimensions as per schema.
            # Current behavior: prints a warning and sets to a 1x1 grid. This is a form of graceful degradation.
            print(f"Warning: resize_grid called with non-positive dimensions ({new_width}x{new_height}). Setting to 1x1 grid.")
            self.grid = np.zeros((1,1), dtype=self.grid.dtype) # Ensure it's at least 1x1
            self.height, self.width = self.grid.shape
            return

        # Create a new grid with the target dimensions, filled with zeros (or current grid's dtype default)
        resized_grid = np.zeros((new_height, new_width), dtype=self.grid.dtype)
        
        # Determine the dimensions of the overlapping region to copy
        copy_height = min(self.height, new_height)
        copy_width = min(self.width, new_width)
        
        # Copy the content from the old grid to the new resized grid
        resized_grid[:copy_height, :copy_width] = self.grid[:copy_height, :copy_width]
        
        # Update the working grid and its dimensions
        self.grid = resized_grid
        self.height, self.width = self.grid.shape

    def copy_selection(self, start_x: int, start_y: int, end_x: int, end_y: int, paste_origins: List[List[int]]) -> None:
        """
        Copies a selected rectangular area from the current grid and pastes it to
        one or more specified locations on the grid.

        The selection is defined by (start_x, start_y) and (end_x, end_y) which
        are inclusive coordinates of any two opposite corners of the selection.
        Pasting is done respecting grid boundaries; parts of the selection that
        would fall outside the grid when pasted are clipped.

        Args:
            start_x (int): The x-coordinate of the first corner of the selection.
            start_y (int): The y-coordinate of the first corner of the selection.
            end_x (int): The x-coordinate of the second corner of the selection.
            end_y (int): The y-coordinate of the second corner of the selection.
            paste_origins (List[List[int]]): A list of [x, y] coordinates, where each
                pair represents the top-left anchor point for pasting the selection.
        """
        # Ensure selection coordinates are ordered correctly (top-left, bottom-right)
        sel_x1, sel_x2 = min(start_x, end_x), max(start_x, end_x)
        sel_y1, sel_y2 = min(start_y, end_y), max(start_y, end_y)

        # Clip selection coordinates to be within the source grid bounds
        # This ensures we only try to copy what's actually available.
        sel_x1 = max(0, min(sel_x1, self.width - 1))
        sel_y1 = max(0, min(sel_y1, self.height - 1))
        sel_x2 = max(0, min(sel_x2, self.width - 1))
        sel_y2 = max(0, min(sel_y2, self.height - 1))

        # If the clipped selection forms an invalid or empty area, there's nothing to copy.
        if sel_x1 > sel_x2 or sel_y1 > sel_y2:
            return

        # Extract the selected portion of the grid
        selection_to_copy = self.grid[sel_y1 : sel_y2 + 1, sel_x1 : sel_x2 + 1].copy()
        selection_height, selection_width = selection_to_copy.shape

        if selection_height == 0 or selection_width == 0:
            return # Nothing to paste if selection is empty

        # Paste the selection to each specified origin
        for paste_x, paste_y in paste_origins:
            # Determine the bounds of the paste area in the target grid
            paste_target_x1 = paste_x
            paste_target_y1 = paste_y
            paste_target_x2 = paste_x + selection_width -1 # inclusive end
            paste_target_y2 = paste_y + selection_height -1 # inclusive end
            
            # Determine the actual part of the selection that will be pasted after clipping
            # (i.e., what part of the selection fits into the target grid at this paste_origin)
            
            # Clip paste_target coordinates to grid boundaries
            final_paste_x1 = max(0, paste_target_x1)
            final_paste_y1 = max(0, paste_target_y1)
            final_paste_x2 = min(self.width - 1, paste_target_x2)
            final_paste_y2 = min(self.height - 1, paste_target_y2)

            # If the final paste area is invalid (e.g., completely outside), skip
            if final_paste_x1 > final_paste_x2 or final_paste_y1 > final_paste_y2:
                continue

            # Determine the corresponding part of the `selection_to_copy` to use
            # This accounts for cases where only a part of the selection can be pasted.
            sel_src_x1 = final_paste_x1 - paste_target_x1
            sel_src_y1 = final_paste_y1 - paste_target_y1
            sel_src_x2 = sel_src_x1 + (final_paste_x2 - final_paste_x1)
            sel_src_y2 = sel_src_y1 + (final_paste_y2 - final_paste_y1)
            
            # Perform the paste operation
            self.grid[final_paste_y1 : final_paste_y2 + 1, final_paste_x1 : final_paste_x2 + 1] = \
                selection_to_copy[sel_src_y1 : sel_src_y2 + 1, sel_src_x1 : sel_src_x2 + 1]

    def get_grid(self) -> List[List[int]]:
        """
        Returns the current state of the working grid as a list of lists of integers.

        Returns:
            List[List[int]]: The current grid.
        """
        return self.grid.tolist()

    def execute_python_code(self, code: str) -> Dict[str, Union[bool, str]]:
        """
        Executes a provided string of Python code.

        The code is executed in a restricted environment that has access to:
        - `self.grid`: The current grid as a NumPy array, which can be modified directly.
        - `self.height`: The height of the current grid.
        - `self.width`: The width of the current grid.
        - `np`: The NumPy library.
        
        Caution: This method uses `exec()`, which can be dangerous if the input `code`
        is not carefully controlled and validated. It is intended for use with code
        generated by a trusted source (e.g., the LLM in this system).

        Args:
            code (str): A string containing the Python code to be executed.
                
        Returns:
            Dict[str, Union[bool, str]]: A dictionary indicating the outcome.
                - "success" (bool): True if the code executed without exceptions, False otherwise.
                - "message" (str): "Code executed successfully" on success, or an
                                   error message if an exception occurred.
        """
        # Define a dictionary of allowed global variables for the exec environment
        # This provides access to the grid and numpy for the executed code.
        allowed_globals = {
            'np': np,    # NumPy library
            'self': self # The GridOperations instance itself, allowing access to self.grid, self.height, self.width
        }
        
        # Execute the provided code string within the defined globals and locals
        try:
            exec(code, allowed_globals)
            # After execution, self.grid might have been modified.
            # Update self.height and self.width in case the code changed grid dimensions.
            self.height, self.width = self.grid.shape
            return {"success": True, "message": "Code executed successfully"}
        except Exception as e:
            # If any error occurs during code execution, catch it and return an error message.
            return {"success": False, "message": f"Error executing Python code: {str(e)}"} 