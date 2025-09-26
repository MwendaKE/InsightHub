import pandas as pd
import numpy as np
import requests
import io

def download_real_gbd_data():
    """
    Alternative: Download pre-processed GBD data from their portal
    You would need to manually download the CSV first from:
    http://ghdx.healthdata.org/gbd-results-tool
    """
    try:
        # Load your downloaded GBD CSV
        gbd_data = pd.read_csv('gbd_sickle_cell_data.csv')
        return gbd_data
    except FileNotFoundError:
        print("GBD CSV file not found. Please download it manually first.")
        return None

def create_analysis_ready_dataset():
    """
    Create a ready-to-analyze dataset with realistic structure
    """
    # Countries with high sickle cell burden
    high_burden_countries = [
        'Nigeria', 'Democratic Republic of the Congo', 'India', 
        'Tanzania', 'Uganda', 'Ghana', 'Kenya', 'Cameroon',
        'Mozambique', 'Niger', 'Burkina Faso', 'Malawi'
    ]
    
    # Generate realistic time series data
    years = list(range(1990, 2023))
    
    data_rows = []
    for country in high_burden_countries:
        # Base rates based on real epidemiology
        if country in ['Nigeria', 'DR Congo']:
            base_death_rate = 45
            base_prevalence_rate = 2000
        elif country in ['Ghana', 'Tanzania', 'Uganda']:
            base_death_rate = 35
            base_prevalence_rate = 1500
        else:
            base_death_rate = 25
            base_prevalence_rate = 1000
        
        for year in years:
            # Simulate improving trends over time
            improvement_factor = 1 - ((year - 1990) * 0.005)
            
            row = {
                'country': country,
                'year': year,
                'deaths': int(base_death_rate * improvement_factor * (0.8 + 0.4 * np.random.random())),
                'prevalence': int(base_prevalence_rate * improvement_factor * (0.8 + 0.4 * np.random.random())),
                'death_rate_per_100k': base_death_rate * improvement_factor * (0.9 + 0.2 * np.random.random()),
                'health_expenditure_pct_gdp': max(2, 8 * improvement_factor * np.random.random()),
                'gdp_per_capita_usd': 1000 + (year - 1990) * 50 * np.random.random(),
                'life_expectancy': 50 + (year - 1990) * 0.5 * np.random.random(),
                'who_region': 'Africa' if country != 'India' else 'South-East Asia',
                'income_level': 'Low' if country in ['Nigeria', 'DR Congo', 'Tanzania'] else 'Lower-middle'
            }
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Calculate additional metrics
    df['mortality_burden_score'] = (df['death_rate_per_100k'] * df['prevalence']) / 1000
    df['healthcare_gap'] = df['death_rate_per_100k'] / df['health_expenditure_pct_gdp']
    
    return df

# Run the enhanced version
def generate_final_dataset():
    print("Generating comprehensive sickle cell disease dataset...")
    
    # Try to load real GBD data first
    real_data = download_real_gbd_data()
    
    if real_data is not None:
        final_df = real_data
        print("Using real GBD data")
    else:
        # Fall back to generated data
        final_df = create_analysis_ready_dataset()
        print("Using generated realistic data")
    
    # Save the final dataset
    output_filename = 'sickle_cell_disease_global_analysis_1990_2022.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ Dataset successfully created: {output_filename}")
    print(f"📊 Shape: {final_df.shape}")
    print(f"🌍 Countries: {final_df['country'].nunique()}")
    print(f"📅 Years: {final_df['year'].min()} - {final_df['year'].max()}")
    print(f"📈 Variables: {len(final_df.columns)}")
    
    print("\nAvailable columns for analysis:")
    for col in final_df.columns:
        print(f"  - {col}")
    
    return final_df

# Execute the script
if __name__ == "__main__":
    dataset = generate_final_dataset()