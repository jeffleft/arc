import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from datetime import datetime
from io import BytesIO
import base64

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_training_runs():
    results_dir = Path("results")
    return sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)

def get_task_samples(task_dir):
    """Get the training samples for a task from the original task data file"""
    task_name = task_dir.name
    task_file = Path("../data/training") / f"{task_name}.json"
    
    if not task_file.exists():
        return []
        
    task_data = load_json(task_file)
    return task_data.get("train", [])

def get_plan(message_history):
    """Get the first assistant message which contains the plan"""
    for msg in message_history:
        if msg.get('role') == 'assistant':
            return msg.get('content', '')
    return ''

def get_tool_calls_with_states(task_dir):
    """Get all tool calls and their corresponding intermediate states"""
    message_history = load_json(task_dir / "message_history.json")
    intermediate_states = load_json(task_dir / "intermediate_states/states.json")
    
    tool_calls = []
    for msg in message_history:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                # Parse the arguments JSON string
                try:
                    args = json.loads(tool_call['function']['arguments'])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                
                tool_calls.append({
                    'tool': tool_call['function']['name'],
                    'args': args,
                    'rationale': args.get('rationale', ''),
                    'state': next((s for s in intermediate_states if s['tool'] == tool_call['function']['name']), None)
                })
    return tool_calls

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
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    st.title("ARC Solver Training Run Analysis")
    
    # Get all training runs
    training_runs = get_training_runs()
    
    # Sidebar for run selection
    selected_run = st.sidebar.selectbox(
        "Select Training Run",
        training_runs,
        format_func=lambda x: x.name
    )
    
    st.header(f"Analysis for {selected_run.name}")
    
    # Load prompt evolution history
    evolution_path = selected_run / "prompt_evolution_history.json"
    if evolution_path.exists():
        evolution_data = load_json(evolution_path)
        
        # Create a dataframe of prompts and results
        prompt_data = []
        for entry in evolution_data:
            prompt_data.append({
                'prompt': entry['prompt'],
                'score': entry['score'],
                'commentary': entry['commentary']
            })
        
        df_prompts = pd.DataFrame(prompt_data)
        
        # Show prompt evolution
        st.subheader("Prompt Evolution")
        for i, row in df_prompts.iterrows():
            with st.expander(f"Task {i+1} - Score: {row['score']}"):
                st.text_area("Prompt", row['prompt'], height=200)
                st.text_area("Analysis", row['commentary'], height=100)
    
    # Get all task directories
    task_dirs = [d for d in selected_run.iterdir() if d.is_dir() and d.name != "intermediate_states"]
    
    # Show task results summary
    st.subheader("Task Results Summary")
    metrics = []
    for task_dir in task_dirs:
        results_path = task_dir / "results.json"
        if results_path.exists():
            results = load_json(results_path)
            metrics.append({
                'task_id': task_dir.name,
                'success': results.get('score', 0),
                'confidence': results.get('confidence', 0),
                'message_count': len(load_json(task_dir / "message_history.json"))
            })
    
    if metrics:
        df = pd.DataFrame(metrics)
        st.dataframe(df)
        
        # Plot success rate
        success_rate = df['success'].mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Show detailed task analysis
    st.subheader("Detailed Task Analysis")
    selected_task = st.selectbox(
        "Select Task",
        task_dirs,
        format_func=lambda x: x.name
    )
    
    if selected_task:
        results = load_json(selected_task / "results.json")
        
        # Show task samples
        st.subheader("Task Samples")
        samples = get_task_samples(selected_task)
        for i, sample in enumerate(samples):
            with st.expander(f"Sample {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    # Convert input grid to image
                    input_grid = sample["input"]
                    input_img = grid_to_image(input_grid)
                    st.image(f"data:image/png;base64,{input_img}", caption="Input")
                with col2:
                    # Convert output grid to image
                    output_grid = sample["output"]
                    output_img = grid_to_image(output_grid)
                    st.image(f"data:image/png;base64,{output_img}", caption="Expected Output")
        
        # Show current task input/output
        st.subheader("Current Task")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(selected_task / "input.png", caption="Input")
        with col2:
            st.image(selected_task / "expected.png", caption="Expected")
        with col3:
            st.image(selected_task / "predicted.png", caption="Predicted")
        
        # Show plan
        message_history = load_json(selected_task / "message_history.json")
        plan = get_plan(message_history)
        st.subheader("Initial Plan")
        st.text_area("Plan", plan, height=150)
        
        # Show tool calls and intermediate states
        st.subheader("Tool Calls and Intermediate States")
        tool_calls = get_tool_calls_with_states(selected_task)
        
        for i, tool_call in enumerate(tool_calls):
            with st.expander(f"Step {i+1}: {tool_call['tool']}"):
                st.write("Rationale:", tool_call['rationale'])
                st.write("Tool Call Args: ", tool_call["args"])
                if tool_call['state']:
                    st.image(selected_task / f"intermediate_states/step_{tool_call['state']['step']:03d}.png", 
                            caption=f"After {tool_call['tool']}")
        
        # Show final analysis
        st.subheader("Final Analysis")
        st.text_area("Analysis", results.get('commentary', ''), height=200)

if __name__ == "__main__":
    main() 