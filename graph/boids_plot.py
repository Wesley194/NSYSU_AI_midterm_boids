import json
import pandas as pd
import matplotlib.pyplot as plt

# Set the file name
file_name = """data/record/record_2025-11-30_15-42-05.json"""
metric_to_plot = ['Size', 'MIN_Speed', 'MAX_Speed', 'Perception_Radius', 'Separation_Weight', 'Alignment_Weight', 'Cohesion_Weight', 'Flee_Weight', 'Alert_Radius', 'Fitness']

def plot_time_series_data_english(file_name, metric_name):
    """Reads the JSON file, processes the data, and plots the specified metric against time."""
    
    # Read the JSON file
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_name}.")
        return None

    if not data:
        print("Error: JSON file contains no data.")
        return None

    # Flatten the nested JSON data, preserving the 'time' column
    df_flat = pd.json_normalize(data, record_path='birds', meta=['time'])

    # Check if the specified metric column exists
    if metric_name not in df_flat.columns:
        available_metrics = [col for col in df_flat.columns if col not in ['time', 'Size']]
        print(f"Error: Metric column '{metric_name}' not found. "
              f"Available columns include: {', '.join(available_metrics)}")
        return None

    # Calculate the initial time for the time baseline (t=0)
    initial_time = df_flat['time'].min()
    
    # Calculate relative time in seconds (X-axis)
    df_flat['relative_time_s'] = (df_flat['time'] - initial_time) / 1000

    # Calculate the mean of the specified metric for each time step (Y-axis data)
    df_summary = df_flat.groupby('relative_time_s')[metric_name].mean().reset_index()
    
    # Plotting the time series trend
    plt.figure(figsize=(10, 6))
    plt.plot(df_summary['relative_time_s'], df_summary[metric_name], marker='o', linestyle='-')
    
    # --- Output fields ---
    plt.title(f"Time Series Trend of Average {metric_name}", fontsize=14)
    plt.xlabel("Relative Time (seconds)", fontsize=12)
    plt.ylabel(f"Average {metric_name}", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(df_summary['relative_time_s'])
    
    # Save the figure to a file
    output_filename = f"graph/average_{metric_name.lower()}_time_series_plot.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Analysis metric: {metric_name}")
    return output_filename

# --- Specify the metric to plot (Example: 'Fitness') ---
for i in metric_to_plot:
    output_file = plot_time_series_data_english(file_name, i)
    if output_file:
        print(f"Plot saved successfully to: {output_file}")