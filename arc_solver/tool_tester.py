import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from grid_ops import GridOperations

from solver import ARCSolver

A = ARCSolver(api_key="")

def grid_to_image(grid):
    """Convert a grid to a base64-encoded PNG image.
    Each tile is 8x8 pixels with 1px white separators."""
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
    TILE_SIZE = 8
    SEPARATOR_WIDTH = 1
    
    # Calculate image dimensions
    height, width = arr.shape
    img_width = width * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH
    img_height = height * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH
    
    # Create RGB image
    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))  # White background
    pixels = img.load()
    
    # Draw tiles
    for y in range(height):
        for x in range(width):
            color = colors.get(arr[y, x], (0, 0, 0))
            
            # Calculate tile position
            tile_x = x * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH
            tile_y = y * (TILE_SIZE + SEPARATOR_WIDTH) + SEPARATOR_WIDTH
            
            # Fill tile
            for ty in range(TILE_SIZE):
                for tx in range(TILE_SIZE):
                    pixels[tile_x + tx, tile_y + ty] = color
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_grid(grid):
    """Display a grid as an image in Streamlit"""
    #img_data = grid_to_image(grid)
    img_data = A._grid_to_image(grid)
    st.image(f"data:image/png;base64,{img_data}")

def main():
    st.title("ARC Grid Tool Tester")
    
    # Initialize grid operations
    if 'grid_ops' not in st.session_state:
        initial_grid = np.zeros((10, 10), dtype=np.uint8)  # Default 10x10 grid
        st.session_state.grid_ops = GridOperations(initial_grid)
    
    # Sidebar for grid size
    st.sidebar.header("Grid Settings")
    grid_size = st.sidebar.slider("Grid Size", 5, 30, 10)
    
    if st.sidebar.button("Reset Grid"):
        initial_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        st.session_state.grid_ops = GridOperations(initial_grid)
    
    # Tool selection
    st.header("Select Tool")
    tool = st.selectbox(
        "Choose a tool to use",
        ["fill_tiles", "fill_pattern", "fill_rectangle", "translate", "resize_grid", "copy_selection"]
    )
    
    # Tool-specific parameters
    if tool == "fill_tiles":
        st.subheader("Fill Tiles")
        positions = st.text_area("Positions (list of [x,y] coordinates)", "[[0,0], [1,1]]")
        color = st.number_input("Color (0-9)", 0, 9, 1)
        if st.button("Apply Fill Tiles"):
            try:
                raw_positions = eval(positions)
                # Convert [x,y] format to {"x": x, "y": y, "color": color} format
                formatted_positions = [{"x": pos[0], "y": pos[1], "color": color} for pos in raw_positions]
                st.session_state.grid_ops.fill_tiles(formatted_positions)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif tool == "fill_pattern":
        st.subheader("Fill Pattern")
        col1, col2 = st.columns(2)
        with col1:
            start_x = st.number_input("Start X", 0, grid_size-1, 0)
            start_y = st.number_input("Start Y", 0, grid_size-1, 0)
            direction = st.selectbox("Direction", ["horizontal", "vertical"])
        with col2:
            interval = st.number_input("Interval", 1, grid_size, 1)
            color = st.number_input("Color (0-9)", 0, 9, 1)
        if st.button("Apply Fill Pattern"):
            st.session_state.grid_ops.fill_pattern(start_x, start_y, direction, interval, color)
    
    elif tool == "fill_rectangle":
        st.subheader("Fill Rectangle")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("X1", 0, grid_size-1, 0)
            y1 = st.number_input("Y1", 0, grid_size-1, 0)
        with col2:
            x2 = st.number_input("X2", 0, grid_size-1, grid_size-1)
            y2 = st.number_input("Y2", 0, grid_size-1, grid_size-1)
        color = st.number_input("Color (0-9)", 0, 9, 1)
        if st.button("Apply Fill Rectangle"):
            st.session_state.grid_ops.fill_rectangle(x1, y1, x2, y2, color)
    
    elif tool == "translate":
        st.subheader("Translate")
        dx = st.number_input("Delta X", -grid_size, grid_size, 0)
        dy = st.number_input("Delta Y", -grid_size, grid_size, 0)
        if st.button("Apply Translate"):
            st.session_state.grid_ops.translate(dx, dy)
    
    elif tool == "resize_grid":
        st.subheader("Resize Grid")
        new_width = st.number_input("New width", 5, 30, grid_size)
        new_height = st.number_input("New height", 5, 30, grid_size)
        if st.button("Apply Resize"):
            st.session_state.grid_ops.resize_grid(new_width, new_height)
    
    elif tool == "copy_selection":
        st.subheader("Copy Selection")
        col1, col2 = st.columns(2)
        with col1:
            start_x = st.number_input("Start X", 0, grid_size-1, 0)
            start_y = st.number_input("Start Y", 0, grid_size-1, 0)
        with col2:
            end_x = st.number_input("End X", 0, grid_size-1, grid_size-1)
            end_y = st.number_input("End Y", 0, grid_size-1, grid_size-1)
        paste_origins = st.text_area("Paste Origins (list of [x,y] coordinates)", "[[0,0]]")
        if st.button("Apply Copy Selection"):
            try:
                paste_origins = eval(paste_origins)
                st.session_state.grid_ops.copy_selection(start_x, start_y, end_x, end_y, paste_origins)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display current grid
    st.header("Current Grid")
    current_grid = st.session_state.grid_ops.get_grid()
    display_grid(current_grid)
    
    # Display grid as array
    st.subheader("Grid Array")
    st.write(current_grid)

if __name__ == "__main__":
    main() 