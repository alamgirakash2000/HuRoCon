import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_velocity():
    # Read data from CSV
    try:
        data = pd.read_csv('velocity_data.csv')
        print(f"Read {len(data)} data points")
    except Exception as e:
        print(f"Error reading velocity data: {e}")
        return
    
    # Define colors for each terrain type
    terrain_colors = {
        "FLAT": "blue",
        "RAMP": "orange",
        "TRENCH": "green",
        "BEAM": "red",
        "STAIRS": "purple",
        "SLIPPERY": "brown",
        "OBSTACLES": "pink"
    }
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each terrain type with different color
    for terrain in data['terrain'].unique():
        terrain_data = data[data['terrain'] == terrain]
        plt.plot(terrain_data['time'], terrain_data['velocity'], 
                 color=terrain_colors.get(terrain, 'gray'),
                 label=terrain)
    
    # Add labels and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Robot Velocity vs Time by Terrain')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('velocity_plot.png', dpi=300)
    print("Plot saved to velocity_plot.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_velocity()
