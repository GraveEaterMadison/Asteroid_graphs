import requests
import pandas as pd
import logging
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


logging.basicConfig(level=config['logging']['level'])
logger = logging.getLogger(__name__)


NASA_API_KEY = config['nasa_api']['key']
ASTEROID_URL = config['nasa_api']['asteroid_url'].format(key=NASA_API_KEY)
EXOPLANET_URL = config['nasa_api']['exoplanet_url']

def fetch_data(api_url):
    """Fetch data from NASA API and handle errors."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        logger.info("Data fetched successfully from NASA API")
        return data
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

def process_data(data, data_type):
    """Process and clean data based on type."""
    if data_type == 'asteroids':
        try:
            if isinstance(data, list):
                asteroids = data
            elif 'near_earth_objects' in data:
                asteroids = data['near_earth_objects']
            else:
                raise ValueError("Unexpected structure of asteroid data")

            rows = []
            if isinstance(asteroids, dict):
                for date, asteroid_list in asteroids.items():
                    for asteroid in asteroid_list:
                        try:
                            name = asteroid['name']
                            diameter = float(asteroid['estimated_diameter']['kilometers']['estimated_diameter_max'])
                            distance = float(asteroid['close_approach_data'][0]['miss_distance']['kilometers']) if 'close_approach_data' in asteroid and len(asteroid['close_approach_data']) > 0 else None
                            distance_ly = distance * 1.057e-17  # Convert km to light years
                            rows.append({'name': name, 'diameter_km': diameter, 'distance_km': distance, 'distance_ly': distance_ly})
                        except KeyError as e:
                            logger.warning(f"Missing data in asteroid entry: {e}")
                        except ValueError as e:
                            logger.warning(f"Data conversion error: {e}")
            elif isinstance(asteroids, list):
                for asteroid in asteroids:
                    try:
                        name = asteroid['name']
                        diameter = float(asteroid['estimated_diameter']['kilometers']['estimated_diameter_max'])
                        distance = float(asteroid['close_approach_data'][0]['miss_distance']['kilometers']) if 'close_approach_data' in asteroid and len(asteroid['close_approach_data']) > 0 else None
                        distance_ly = distance * 1.057e-17  # Convert km to light years
                        rows.append({'name': name, 'diameter_km': diameter, 'distance_km': distance, 'distance_ly': distance_ly})
                    except KeyError as e:
                        logger.warning(f"Missing data in asteroid entry: {e}")
                    except ValueError as e:
                        logger.warning(f"Data conversion error: {e}")

            df = pd.DataFrame(rows)

        except KeyError as e:
            logger.error(f"KeyError: {e}")
            raise

    elif data_type == 'exoplanets':
        try:
            df = pd.json_normalize(data['bodies'])  # Adjust based on actual data structure
            df = df[['name', 'distance', 'radius']]  # Adjust based on actual data fields
            df.columns = ['name', 'distance_light_years', 'radius_earth_radii']
            df['distance_light_years'] = pd.to_numeric(df['distance_light_years'], errors='coerce')
            df['radius_earth_radii'] = pd.to_numeric(df['radius_earth_radii'], errors='coerce')
        except KeyError as e:
            logger.error(f"KeyError: {e}")
            raise

    else:
        raise ValueError("Invalid data type")

    # Validate and clean the data
    if df.empty:
        raise ValueError("The dataframe is empty. No data to process.")
    if any(df.isna().sum()):
        logger.warning("Data contains missing values. Cleaning data.")
        df = df.dropna()  # Example of handling missing values

    return df

def load_data(data_type):
    """Load data based on user choice."""
    if data_type == 'asteroids':
        url = ASTEROID_URL
    elif data_type == 'exoplanets':
        url = EXOPLANET_URL
    else:
        raise ValueError("Invalid data type")
    
    data = fetch_data(url)
    df = process_data(data, data_type)
    return df

def plot_data(df, data_type):
    """Plot data based on data type with advanced features."""
    plt.figure(figsize=(14, 7))
    
    # Create a dictionary for temporary labels
    label_dict = {}
    labels = []

    if data_type == 'asteroids':
        # Assign temporary labels to each point
        for i, (idx, row) in enumerate(df.iterrows()):
            label = f'p{i+1}'
            label_dict[label] = row['name']
            labels.append(label)

        # Plot with temporary labels
        scatter = sns.scatterplot(data=df, x='diameter_km', y='distance_ly', hue='name', palette='viridis', legend=None)

        plt.title('Asteroids: Diameter vs Distance')
        plt.xlabel('Diameter (km)')
        plt.ylabel('Distance (light years)')

        for i, label in enumerate(labels):
            scatter.text(df.loc[i, 'diameter_km'], df.loc[i, 'distance_ly'], label, fontsize=8, ha='right')

        # Add a separate legend with actual names
        plt.figtext(0.99, 0.99, 'Legend:', horizontalalignment='right', verticalalignment='top', fontsize=10, color='black')
        for i, (label, name) in enumerate(label_dict.items()):
            plt.figtext(0.99, 0.95 - i*0.02, f'{label}: {name}', horizontalalignment='right', verticalalignment='top', fontsize=8, color='black')

    elif data_type == 'exoplanets':
        # Assign temporary labels to each point
        for i, (idx, row) in enumerate(df.iterrows()):
            label = f'p{i+1}'
            label_dict[label] = row['name']
            labels.append(label)

        # Plot with temporary labels
        scatter = sns.scatterplot(data=df, x='distance_light_years', y='radius_earth_radii', hue='name', palette='viridis', legend=None)

        plt.title('Exoplanets: Distance vs Radius')
        plt.xlabel('Distance (light years)')
        plt.ylabel('Radius (Earth radii)')

        for i, label in enumerate(labels):
            scatter.text(df.loc[i, 'distance_light_years'], df.loc[i, 'radius_earth_radii'], label, fontsize=8, ha='right')

        # Add a separate legend with actual names
        plt.figtext(0.99, 0.99, 'Legend:', horizontalalignment='right', verticalalignment='top', fontsize=10, color='black')
        for i, (label, name) in enumerate(label_dict.items()):
            plt.figtext(0.99, 0.95 - i*0.02, f'{label}: {name}', horizontalalignment='right', verticalalignment='top', fontsize=8, color='black')

    plt.figtext(0.99, 0.01, '© Sayak Kundu', horizontalalignment='right', verticalalignment='bottom', fontsize=8, color='gray')
    plt.tight_layout()
    
    # Add interactive zoom functionality
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(f'{data_type}_data_plot.png')  # Save the plot
    plt.show()

if __name__ == "__main__":
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # User input for data type
    data_type = input("Enter the type of data to load (asteroids/exoplanets): ").strip().lower()
    print(f"Attempting to load {data_type} data from NASA API")

    try:
        df = load_data(data_type)
        print(df.head())
        plot_data(df, data_type)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


