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
    #task_file = Path("../data/v2/evaluation") / f"{task_name}.json"
    
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
    
    print(f"Processing message history with {len(message_history)} messages")
    print(f"Found {len(intermediate_states)} intermediate states")
    
    tool_calls = []
    for msg_idx, msg in enumerate(message_history):
        print(f"\nProcessing message {msg_idx + 1}:")
        print(f"Type: {msg.get('type')}")
        print(f"Role: {msg.get('role')}")
        
        # Handle function calls
        if msg.get('type') == 'function_call':
            print(f"Processing function call: {msg.get('name')}")
            try:
                args = json.loads(msg['arguments'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing function call arguments: {e}")
                args = {}
            
            # Find matching state
            matching_state = None
            for state in intermediate_states:
                if (state.get('tool') == msg['name'] and 
                    state.get('tool_call_id') == msg.get('call_id')):
                    matching_state = state
                    break
            
            tool_calls.append({
                'tool': msg['name'],
                'tool_call_id': msg.get('call_id'),
                'args': args,
                'rationale': args.get('rationale', ''),
                'state': matching_state,
                'message_idx': msg_idx
            })
        
        # Handle assistant messages
        elif msg.get('role') == 'assistant':
            if msg.get('content'):
                if isinstance(msg['content'], list):
                    for content_item in msg['content']:
                        if content_item.get('type') == 'text':
                            print(f"Found text content: {content_item.get('text')[:100]}...")
                else:
                    print(f"Found content: {msg['content'][:100]}...")
    
    print(f"\nTotal tool calls found: {len(tool_calls)}")
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
                'input_tokens': results.get('token_usage', {}).get('input_tokens', 0),
                'output_tokens': results.get('token_usage', {}).get('output_tokens', 0),
                'total_tokens': results.get('token_usage', {}).get('total_tokens', 0),
                'message_count': len(load_json(task_dir / "message_history.json"))
            })
    
    if metrics:
        df = pd.DataFrame(metrics)
        # Format token columns as integers
        df['input_tokens'] = df['input_tokens'].astype(int)
        df['output_tokens'] = df['output_tokens'].astype(int)
        df['total_tokens'] = df['total_tokens'].astype(int)
        
        # Calculate costs
        INPUT_COST_PER_TOKEN = 1.10 / 1_000_000  # $1.10 per 1M tokens
        OUTPUT_COST_PER_TOKEN = 4.40 / 1_000_000  # $4.40 per 1M tokens
        
        input_cost = df['input_tokens'].sum() * INPUT_COST_PER_TOKEN
        output_cost = df['output_tokens'].sum() * OUTPUT_COST_PER_TOKEN
        total_cost = input_cost + output_cost
        
        # Display only token counts in the table
        display_columns = ['task_id', 'success', 'confidence', 'input_tokens', 'output_tokens', 'total_tokens', 'message_count']
        st.dataframe(df[display_columns])
        
        # Plot success rate
        success_rate = df['success'].mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Show average cost per task
        avg_cost = total_cost / len(df)
        st.metric("Average Cost per Task", f"${avg_cost:.4f}")
        
        # Show total cost
        st.metric("Total Cost", f"${total_cost:.4f}")
    
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
            st.image(str(selected_task / "input.png"), caption="Input")
        with col2:
            st.image(str(selected_task / "expected.png"), caption="Expected")
        with col3:
            st.image(str(selected_task / "predicted.png"), caption="Predicted")
        
        # Show plan
        message_history = load_json(selected_task / "message_history.json")
        plan = get_plan(message_history)
        st.subheader("Initial Plan")
        st.text_area("Plan", plan, height=150)
        
        # Show tool calls and intermediate states
        st.subheader("Tool Calls and Intermediate States")
        tool_calls = get_tool_calls_with_states(selected_task)
        message_history = load_json(selected_task / "message_history.json")
        
        # Load intermediate states
        intermediate_states = []
        states_path = selected_task / "intermediate_states/states.json"
        if states_path.exists():
            intermediate_states = load_json(states_path)
        
        # Create a combined timeline of messages and tool calls
        timeline = []
        step_counter = 0
        
        # Add all messages to timeline
        for msg_idx, msg in enumerate(message_history):
            if msg.get('type') == 'function_call':
                timeline.append({
                    'type': 'tool_call',
                    'data': msg,
                    'message_idx': msg_idx,
                    'step': step_counter
                })
            elif msg.get('role') == 'assistant' and msg.get('content'):
                timeline.append({
                    'type': 'message',
                    'data': msg,
                    'message_idx': msg_idx,
                    'step': step_counter
                })
            elif msg.get('role') == 'user' and msg.get('content'):
                # Check if this is a message containing a grid state
                if isinstance(msg['content'], list):
                    for content_item in msg['content']:
                        if (content_item.get('type') == 'input_text' and 
                            'Current output grid state:' in content_item.get('text', '')):
                            step_counter += 1
                            timeline.append({
                                'type': 'grid_state',
                                'data': msg,
                                'message_idx': msg_idx,
                                'step': step_counter
                            })
                            break
        
        # Sort timeline by message index
        timeline.sort(key=lambda x: x['message_idx'])
        
        # Display timeline
        for i, item in enumerate(timeline):
            if item['type'] == 'tool_call':
                msg = item['data']
                with st.expander(f"Step {item['step']}: {msg['name']}"):
                    try:
                        args = json.loads(msg['arguments'])
                        st.write("Tool Call ID:", msg.get('call_id'))
                        st.write("Rationale:", args.get('rationale', ''))
                        
                        # Pretty print code if present in arguments
                        if 'code' in args:
                            st.write("Arguments:")
                            st.code(args['code'], language='python')
                            # Remove code from args for display
                            display_args = args.copy()
                            del display_args['code']
                            if display_args:
                                st.write("Additional arguments:", json.dumps(display_args, indent=2))
                        else:
                            st.write("Arguments:", json.dumps(args, indent=2))
                    except json.JSONDecodeError:
                        st.write("Raw arguments:", msg['arguments'])
                    
                    # Find the next grid state in the timeline
                    next_grid_state = None
                    for future_item in timeline[i+1:]:
                        if future_item['type'] == 'grid_state':
                            next_grid_state = future_item
                            break
                    
                    if next_grid_state:
                        # Extract grid from the message
                        grid_text = None
                        for content_item in next_grid_state['data']['content']:
                            if (content_item.get('type') == 'input_text' and 
                                'Current output grid state:' in content_item.get('text', '')):
                                grid_text = content_item['text'].split('Current output grid state:\n')[1]
                                break
                        
                        if grid_text:
                            try:
                                grid = json.loads(grid_text)
                                grid_img = grid_to_image(grid)
                                st.image(f"data:image/png;base64,{grid_img}", caption=f"After {msg['name']}")
                            except json.JSONDecodeError:
                                st.warning("Could not parse grid state")
                    else:
                        st.info("No grid state available for this step")
            
            elif item['type'] == 'message':
                msg = item['data']
                with st.expander(f"Step {item['step']}: Assistant Message"):
                    if isinstance(msg['content'], list):
                        for content_item in msg['content']:
                            if content_item.get('type') == 'text':
                                st.write(content_item.get('text', ''))
                    else:
                        st.write(msg.get('content', ''))
        
        # Show final analysis
        st.subheader("Final Analysis")
        st.text_area("Analysis", results.get('commentary', ''), height=200)

if __name__ == "__main__":
    main() 