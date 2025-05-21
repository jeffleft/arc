from typing import List, Dict, Tuple
import numpy as np

class GridOperations:
    def __init__(self, grid: List[List[int]]):
        """Initialize with a grid."""
        self.grid = np.array(grid)
        self.height, self.width = self.grid.shape
        self.dynamic_tools: Dict[str, Callable] = {}
        
    def register_dynamic_tool(self, tool_name: str, tool_code_str: str) -> None:
        """Register a dynamic tool from a code string."""
        try:
            local_scope = {}
            # Add numpy to the scope for the dynamic tool
            exec_globals = {"np": np}
            exec(tool_code_str, exec_globals, local_scope)
            
            if 'dynamic_tool_function' in local_scope:
                self.dynamic_tools[tool_name] = local_scope['dynamic_tool_function']
            else:
                # Fallback: try to find a function with the same name as the tool
                if tool_name in local_scope and callable(local_scope[tool_name]):
                    self.dynamic_tools[tool_name] = local_scope[tool_name]
                else:
                    print(f"Error: 'dynamic_tool_function' or function '{tool_name}' not found in tool code for {tool_name}.")
        except Exception as e:
            print(f"Error registering dynamic tool {tool_name}: {e}")

    def execute_dynamic_tool(self, tool_name: str, **kwargs) -> None:
        """Execute a registered dynamic tool."""
        if tool_name in self.dynamic_tools:
            try:
                self.dynamic_tools[tool_name](self, **kwargs)
            except Exception as e:
                print(f"Error executing dynamic tool {tool_name}: {e}")
        else:
            print(f"Error: Dynamic tool {tool_name} not found.")
            # Potentially raise an error here, e.g., raise ValueError(f"Dynamic tool {tool_name} not found.")

    def fill_tiles(self, positions: List[Dict[str, int]]) -> None:
        """Fill specific tiles with given colors."""
        for pos in positions:
            x, y = pos["x"], pos["y"]
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = pos["color"]
                
    def copy_grid(self) -> None:
        """Copy the input grid to the output (no-op as we're already working on the output)."""
        pass
        
    def fill_pattern(self, start_x: int, start_y: int, direction: str, interval: int, color: int) -> None:
        """Fill tiles in a pattern with fixed interval and direction."""
        if direction == "horizontal":
            for x in range(start_x, self.width, interval):
                if 0 <= x < self.width and 0 <= start_y < self.height:
                    self.grid[start_y, x] = color
        else:  # vertical
            for y in range(start_y, self.height, interval):
                if 0 <= start_x < self.width and 0 <= y < self.height:
                    self.grid[y, start_x] = color
                    
    def fill_rectangle(self, x1: int, y1: int, x2: int, y2: int, color: int) -> None:
        """Fill a rectangle with a given color."""
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        self.grid[y1:y2+1, x1:x2+1] = color
        
    def translate(self, dx: int, dy: int) -> None:
        """Translate the grid by a given offset."""
        # Create a new grid filled with zeros
        new_grid = np.zeros_like(self.grid)
        
        # Calculate the translation bounds
        x_start = max(0, dx)
        x_end = min(self.width, self.width + dx)
        y_start = max(0, dy)
        y_end = min(self.height, self.height + dy)
        
        # Calculate the source bounds
        src_x_start = max(0, -dx)
        src_x_end = min(self.width, self.width - dx)
        src_y_start = max(0, -dy)
        src_y_end = min(self.height, self.height - dy)
        
        # Copy the translated portion
        new_grid[y_start:y_end, x_start:x_end] = self.grid[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Update the grid
        self.grid = new_grid
        
    def resize_grid(self, width: int, height: int) -> None:
        """Resize the grid to MxN dimensions.
        
        Args:
            width: The new width of the grid
            height: The new height of the grid
        """
        # Create a new grid filled with zeros
        new_grid = np.zeros((height, width), dtype=self.grid.dtype)
        
        # Copy the overlapping portion
        copy_height = min(self.height, height)
        copy_width = min(self.width, width)
        
        new_grid[:copy_height, :copy_width] = self.grid[:copy_height, :copy_width]
        
        # Update the grid
        self.grid = new_grid
        self.height, self.width = self.grid.shape

    def copy_selection(self, start_x: int, start_y: int, end_x: int, end_y: int, paste_origins: List[List[int]]) -> None:
        """Copy a selected area to multiple places on the output grid.
        
        Args:
            start_x: Starting x coordinate of selection
            start_y: Starting y coordinate of selection
            end_x: Ending x coordinate of selection
            end_y: Ending y coordinate of selection
            paste_origins: List of [x, y] coordinates where the selection should be pasted
        """
        # Ensure coordinates are ordered correctly
        start_x, end_x = min(start_x, end_x), max(start_x, end_x)
        start_y, end_y = min(start_y, end_y), max(start_y, end_y)
        
        # Ensure selection is within bounds
        start_x = max(0, min(start_x, self.width - 1))
        end_x = max(0, min(end_x, self.width - 1))
        start_y = max(0, min(start_y, self.height - 1))
        end_y = max(0, min(end_y, self.height - 1))
        
        # Extract the selection
        selection = self.grid[start_y:end_y+1, start_x:end_x+1].copy()
        selection_height, selection_width = selection.shape
        
        # Paste to each specified location
        for paste_x, paste_y in paste_origins:
            # Calculate the paste region bounds
            paste_end_x = paste_x + selection_width
            paste_end_y = paste_y + selection_height
            
            # Skip if paste region is completely outside grid
            if (paste_x >= self.width or paste_end_x <= 0 or 
                paste_y >= self.height or paste_end_y <= 0):
                continue
                
            # Calculate the valid paste region
            valid_start_x = max(0, paste_x)
            valid_end_x = min(self.width, paste_end_x)
            valid_start_y = max(0, paste_y)
            valid_end_y = min(self.height, paste_end_y)
            
            # Calculate the corresponding selection region
            sel_start_x = valid_start_x - paste_x
            sel_end_x = sel_start_x + (valid_end_x - valid_start_x)
            sel_start_y = valid_start_y - paste_y
            sel_end_y = sel_start_y + (valid_end_y - valid_start_y)
            
            # Perform the paste
            self.grid[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = \
                selection[sel_start_y:sel_end_y, sel_start_x:sel_end_x]

    def get_grid(self) -> List[List[int]]:
        """Get the current grid state."""
        return self.grid.tolist() 