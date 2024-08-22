import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import data_analysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_distribution(df, column):
    """Plot the distribution of a given column with KDE and handle edge cases."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=20, color='skyblue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_scatter(df, x_column, y_column):
    """Plot a scatter plot with zoom functionality and non-overlapping labels."""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    scatter_plot = sns.scatterplot(data=df, x=x_column, y=y_column, hue=x_column, palette='viridis', legend=None)
    
    # Annotate points with names
    for i in range(df.shape[0]):
        scatter_plot.text(
            df[x_column].iloc[i], df[y_column].iloc[i], df['name'].iloc[i],
            fontsize=8, ha='right', va='bottom', alpha=0.7
        )

    # Adjust axis limits to handle zoom
    scatter_plot.set_xlim(df[x_column].min() - 0.1, df[x_column].max() + 0.1)
    scatter_plot.set_ylim(df[y_column].min() - 0.1, df[y_column].max() + 0.1)

    plt.title(f"{y_column} vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=x_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add copyright note
    plt.figtext(0.99, 0.01, '© Sayak Kundu', horizontalalignment='right', verticalalignment='bottom', fontsize=10)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def interactive_plot(df, x_column, y_column):
    """Create an interactive plot with zoom and pan capabilities."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with interactive zoom
    scatter_plot = ax.scatter(df[x_column], df[y_column], c=df[x_column], cmap='viridis', label=df['name'], alpha=0.7)
    
    # Add labels
    for i in range(df.shape[0]):
        ax.text(
            df[x_column].iloc[i], df[y_column].iloc[i], df['name'].iloc[i],
            fontsize=8, ha='right', va='bottom', alpha=0.7
        )
    
    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    plt.colorbar(scatter_plot, label=x_column)

    # Add grid
    ax.grid(True)

    # Add a cursor for interactive zoom and pan
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    
    # Add copyright note
    plt.figtext(0.99, 0.01, '© Sayak Kundu', horizontalalignment='right', verticalalignment='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_type = input("Enter the type of data to plot (asteroids/exoplanets): ").strip().lower()
    df = data_analysis.load_data(data_type)
    
    if data_type == 'asteroids':
        plot_distribution(df, 'diameter_km')
        plot_distribution(df, 'distance_km')
        interactive_plot(df, 'diameter_km', 'distance_km')
    
    elif data_type == 'exoplanets':
        plot_distribution(df, 'radius_earth_radii')
        plot_distribution(df, 'distance_light_years')
        interactive_plot(df, 'distance_light_years', 'radius_earth_radii')
    
    else:
        raise ValueError("Invalid data type")
