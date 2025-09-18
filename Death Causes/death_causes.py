import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------------------------------
# Data Loading Function
# -------------------------------
def load_causes_of_death_data():
    """Load causes of death data from alternative source"""
    try:
        # Alternative approach: Use OWID's exported data
        # This is a known working dataset from Our World in Data
        url = "https://nyc3.digitaloceanspaces.com/owid-public/data/causes-of-death/causes_of_death.csv"
        
        df = pd.read_csv(url)
        print("✅ Data loaded successfully from alternative source")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Years: {df['year'].min()} to {df['year'].max()}")
        print(f"🌍 Countries: {df['entity'].nunique()}")
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'entity': 'Entity',
            'year': 'Year',
            'code': 'Code'
        })
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading from alternative source: {e}")
        print("🔄 Trying fallback method with sample data...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration if online sources fail"""
    print("📋 Creating sample data for demonstration...")
    
    # Create sample data for analysis
    years = list(range(1990, 2020))
    countries = ['United States', 'United Kingdom', 'Germany', 'Japan', 'Brazil', 'India', 'China', 'South Africa']
    
    data = []
    for year in years:
        for country in countries:
            data.append({
                'Entity': country,
                'Year': year,
                'Code': country[:3].upper(),
                'Deaths - Cardiovascular diseases - Sex: Both - Age: All Ages (Rate)': np.random.uniform(100, 400),
                'Deaths - Neoplasms - Sex: Both - Age: All Ages (Rate)': np.random.uniform(80, 300),
                'Deaths - Chronic respiratory diseases - Sex: Both - Age: All Ages (Rate)': np.random.uniform(20, 100),
                'Deaths - Lower respiratory infections - Sex: Both - Age: All Ages (Rate)': np.random.uniform(10, 80),
                'Deaths - Diabetes mellitus - Sex: Both - Age: All Ages (Rate)': np.random.uniform(5, 50)
            })
    
    df = pd.DataFrame(data)
    print("✅ Sample data created successfully")
    return df

# -------------------------------
# Data Processing Functions
# -------------------------------
def filter_recent_data(df, year=2019):
    """Filter data for the most recent year available"""
    recent_df = df[df['Year'] == year].copy()
    return recent_df

def get_death_rate_columns(df):
    """Identify columns containing death rate data"""
    rate_columns = [col for col in df.columns if 'Rate' in col and 'Deaths' in col]
    return rate_columns

def process_data_for_analysis(df, year=2019):
    """Process data for analysis"""
    # Filter for the specified year
    recent_data = filter_recent_data(df, year)
    
    # Get death rate columns
    rate_columns = get_death_rate_columns(recent_data)
    
    # Create a melted dataframe for easier analysis
    melted_df = recent_data.melt(
        id_vars=['Entity', 'Code', 'Year'], 
        value_vars=rate_columns,
        var_name='Cause_of_Death',
        value_name='Death_Rate'
    )
    
    # Clean cause names
    melted_df['Cause_of_Death'] = melted_df['Cause_of_Death'].str.replace(
        'Deaths - ', '').str.replace(' - Sex: Both - Age: All Ages (Rate)', '')
    
    return melted_df, recent_data

# -------------------------------
# Analysis Functions (Fixed)
# -------------------------------
def analyze_global_causes(melted_df, top_n=15):
    """Analyze global causes of death"""
    global_avg = melted_df.groupby('Cause_of_Death')['Death_Rate'].mean().reset_index()
    global_avg = global_avg.sort_values('Death_Rate', ascending=False).head(top_n)
    
    return global_avg

def analyze_regional_trends(df, causes_of_interest):
    """Analyze regional trends for specific causes"""
    # Define regions
    regions = {
        'Europe': ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland', 'Netherlands'],
        'North America': ['United States', 'Canada', 'Mexico'],
        'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
        'East Asia': ['Japan', 'South Korea', 'China'],
        'South Asia': ['India', 'Pakistan', 'Bangladesh'],
        'Africa': ['South Africa', 'Egypt', 'Kenya', 'Nigeria']
    }
    
    regional_data = {}
    
    for region, countries in regions.items():
        region_df = df[df['Entity'].isin(countries)]
        if not region_df.empty:
            regional_data[region] = {}
            for cause in causes_of_interest:
                col_name = f"Deaths - {cause} - Sex: Both - Age: All Ages (Rate)"
                if col_name in region_df.columns:
                    # Calculate average, handling NaN values
                    avg_value = region_df[col_name].mean()
                    if not pd.isna(avg_value):
                        regional_data[region][cause] = avg_value
    
    return regional_data

def analyze_temporal_trends(df, causes_of_interest, countries_of_interest):
    """Analyze how death rates have changed over time"""
    temporal_data = {}
    
    for cause in causes_of_interest:
        col_name = f"Deaths - {cause} - Sex: Both - Age: All Ages (Rate)"
        if col_name in df.columns:
            temporal_data[cause] = {}
            for country in countries_of_interest:
                country_data = df[df['Entity'] == country]
                if not country_data.empty and col_name in country_data.columns:
                    # Ensure we have data for this country and cause
                    country_cause_data = country_data[['Year', col_name]].dropna()
                    if not country_cause_data.empty:
                        temporal_data[cause][country] = country_cause_data.set_index('Year')[col_name]
    
    return temporal_data

def generate_statistics(df, global_avg):
    """Generate comprehensive statistics for the report"""
    stats = {
        'total_countries': df['Entity'].nunique(),
        'years_covered': f"{df['Year'].min()} - {df['Year'].max()}",
        'global_death_rate_current': global_avg['Death_Rate'].sum(),
        'top_cause': global_avg.iloc[0]['Cause_of_Death'],
        'top_cause_rate': global_avg.iloc[0]['Death_Rate'],
        'second_cause': global_avg.iloc[1]['Cause_of_Death'],
        'second_cause_rate': global_avg.iloc[1]['Death_Rate'],
        'third_cause': global_avg.iloc[2]['Cause_of_Death'],
        'third_cause_rate': global_avg.iloc[2]['Death_Rate']
    }
    return stats

# -------------------------------
# Visualization Functions (Fixed)
# -------------------------------
def plot_global_causes(global_avg, year=2019):
    """Plot global causes of death"""
    plt.figure(figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(global_avg)))
    
    bars = plt.barh(global_avg['Cause_of_Death'], global_avg['Death_Rate'], color=colors)
    
    plt.xlabel('Death Rate (per 100,000 people)', fontsize=12)
    plt.title(f'Top {len(global_avg)} Global Causes of Death ({year})', fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("global_causes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved global causes plot")
    return global_avg

def plot_regional_comparison(regional_data, cause):
    """Plot regional comparison for a specific cause"""
    # Filter out regions that don't have data for this cause
    regions_with_data = []
    values = []
    
    for region, causes in regional_data.items():
        if cause in causes and not pd.isna(causes[cause]):
            regions_with_data.append(region)
            values.append(causes[cause])
    
    if not regions_with_data:
        print(f"⚠️ No data available for {cause} in any region")
        return False
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions_with_data)))
    
    bars = plt.bar(regions_with_data, values, color=colors)
    
    plt.ylabel('Death Rate (per 100,000 people)', fontsize=12)
    plt.title(f'Death Rates from {cause} by Region (2019)', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    filename = f"regional_{cause.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved regional plot: {filename}")
    return True

def plot_temporal_trends(temporal_data, cause, countries):
    """Plot temporal trends for a specific cause"""
    if cause not in temporal_data or not temporal_data[cause]:
        print(f"⚠️ No temporal data available for {cause}")
        return False
    
    plt.figure(figsize=(12, 8))
    has_data = False
    
    for country in countries:
        if country in temporal_data[cause]:
            data = temporal_data[cause][country]
            if not data.empty:
                plt.plot(data.index, data.values, marker='o', linewidth=2, label=country, markersize=4)
                has_data = True
    
    if not has_data:
        print(f"⚠️ No valid temporal data to plot for {cause}")
        plt.close()
        return False
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Death Rate (per 100,000 people)', fontsize=12)
    plt.title(f'Trend in Death Rates from {cause} (1990-2019)', fontsize=16, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"trend_{cause.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved trend plot: {filename}")
    return True

# -------------------------------
# PDF Report Generation (Fixed)
# -------------------------------
def generate_pdf_report(global_avg, regional_data, temporal_data, stats, filename="Global_Causes_of_Death_Analysis.pdf"):
    """Generate comprehensive PDF report with proper pagination and footer"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Colors
    primary_color = HexColor('#2E86AB')
    secondary_color = HexColor('#A23B72')
    accent_color = HexColor('#F18F01')
    
    def add_footer():
        """Add footer to current page"""
        c.setFillColor(HexColor('#666666'))
        c.setFont("Helvetica-Oblique", 8)
        footer_text = "Generated by Insight Hub Analysis Program created by Mwenda E. Njagi at GitHub.com. Link: https://github.com/MwendaKE/InsightHub."
        c.drawCentredString(width/2, 20, footer_text)
    
    def safe_draw_image(image_path, x, y, width_img, height_img):
        """Safely draw image if it exists"""
        if os.path.exists(image_path):
            c.drawImage(image_path, x, y, width_img, height_img)
            return True
        else:
            # Draw placeholder text if image doesn't exist
            c.setFillColor(HexColor('#CCCCCC'))
            c.rect(x, y, width_img, height_img, fill=1)
            c.setFillColor(HexColor('#666666'))
            c.setFont("Helvetica", 10)
            c.drawCentredString(x + width_img/2, y + height_img/2, f"Image not available: {os.path.basename(image_path)}")
            return False
    
    # Title Page
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-100, "GLOBAL CAUSES OF DEATH ANALYSIS REPORT")
    
    c.setFillColor(secondary_color)
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-150, "Comprehensive Mortality Patterns Analysis")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 12)
    c.drawCentredString(width/2, height-200, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 12)
    c.drawCentredString(width/2, height-250, f"Created by Mwenda E. Njagi @ Github.com/MwendaKE/InsightHub")
    
    c.setFillColor(HexColor('#666666'))
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height-300, "Data Source: Sample Data (Real data unavailable)")
    
    # Add some decorative elements
    c.setStrokeColor(primary_color)
    c.setLineWidth(1)
    c.line(50, height-330, width-50, height-330)
    
    add_footer()  # Add footer to first page
    c.showPage()  # End of page 1
    
    # Executive Summary
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-50, "Executive Summary")
    
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica", 10)
    summary_text = [
        f"• Analyzed causes of death data from {stats['total_countries']} countries",
        f"• Coverage period: {stats['years_covered']}",
        f"• Current global death rate: {stats['global_death_rate_current']:.1f} per 100,000 people",
        f"• Leading cause: {stats['top_cause']} ({stats['top_cause_rate']:.1f} per 100k)",
        f"• Second leading cause: {stats['second_cause']} ({stats['second_cause_rate']:.1f} per 100k)",
        "",
        "Note: This report was generated using sample data as",
        "real-world data sources were temporarily unavailable.",
        "The analysis demonstrates the capability of the system",
        "to process and visualize mortality data effectively."
    ]
    
    y_pos = height-80
    for line in summary_text:
        c.drawString(70, y_pos, line)
        y_pos -= 20
    
    # Global Causes
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_pos-40, "Global Causes of Death (2019)")
    safe_draw_image("global_causes.png", 50, y_pos-270, 500, 200)
    
    # Analysis Text
    analysis_text = [
        "Global mortality patterns show a clear epidemiological transition:",
        "• Non-communicable diseases account for the majority of deaths worldwide",
        "• Cardiovascular diseases remain the leading cause of mortality globally",
        "• Neoplasms (cancers) represent the second leading cause of death",
        "• Communicable diseases have declined but remain significant in some regions",
        "",
        "This pattern reflects global development, aging populations, and",
        "the success of public health interventions against infectious diseases."
    ]
    
    y_text = y_pos - 290
    for line in analysis_text:
        c.drawString(70, y_text, line)
        y_text -= 15
    
    add_footer()  # Add footer to second page
    c.showPage()  # End of page 2
    
    # Regional Variations - Page 3
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Regional Variations in Cardiovascular Diseases")
    
    regional_image = "regional_cardiovascular_diseases.png"
    if not safe_draw_image(regional_image, 50, height-250, 500, 180):
        # Add explanation if image not available
        c.setFillColor(HexColor('#666666'))
        c.setFont("Helvetica", 10)
        c.drawString(70, height-450, "Regional analysis visualization could not be generated due to data limitations.")
        c.drawString(70, height-465, "This section would typically show variations in disease prevalence across different regions.")
    
    # Regional Analysis
    regional_analysis = [
        "Cardiovascular disease rates vary significantly by region:",
        "• Eastern Europe typically shows the highest rates",
        "• Western nations have moderate rates despite aging populations",
        "• Developing regions show increasing rates with urbanization",
        "• Some regions show success in reducing cardiovascular mortality",
        "",
        "Factors influencing regional variations include:",
        "• Dietary patterns and salt consumption",
        "• Smoking prevalence and tobacco control policies",
        "• Access to healthcare and preventive services"
    ]
    
    y_regional = height-450
    for line in regional_analysis:
        c.drawString(70, y_regional, line)
        y_regional -= 15
    
    add_footer()  # Add footer to third page
    c.showPage()  # End of page 3
    
    # Key Insights - Page 4
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Key Insights and Recommendations")
    
    insights = [
        "1. DATA AVAILABILITY:",
        "   • Real-world mortality data is crucial for accurate analysis",
        "   • Multiple data sources should be integrated for robustness",
        "   • Regular updates ensure timely insights",
        "",
        "2. SYSTEM CAPABILITIES:",
        "   • This analysis system can process complex mortality data",
        "   • Automated report generation saves time and resources",
        "   • Visualizations help communicate complex patterns effectively",
        "",
        "3. FUTURE ENHANCEMENTS:",
        "   • Integrate with additional data sources when available",
        "   • Add more sophisticated statistical analyses",
        "   • Include predictive modeling capabilities",
        "",
        "4. PUBLIC HEALTH IMPLICATIONS:",
        "   • Understanding mortality patterns informs health policy",
        "   • Regional disparities highlight areas needing intervention",
        "   • Temporal trends help evaluate public health initiatives"
    ]
    
    y_insights = height-80
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica", 10)
    for line in insights:
        c.drawString(70, y_insights, line)
        y_insights -= 15
        if y_insights < 50:  # Handle text overflow
            add_footer()
            c.showPage()
            y_insights = height-50
            c.setFont("Helvetica", 10)
    
    add_footer()  # Add footer to final page
    
    c.save()
    print(f"✅ Comprehensive PDF report generated: {filename}")

# -------------------------------
# Main Analysis Function (Updated)
# -------------------------------
def main():
    print("🚀 Starting Global Causes of Death Analysis...")
    
    # Load data
    print("📊 Loading data from Our World in Data...")
    df = load_causes_of_death_data()
    
    if df.empty:
        print("❌ Error: Could not load data")
        return
    
    # Process data
    print("🔧 Processing data for analysis...")
    melted_df, recent_data = process_data_for_analysis(df)
    
    # Analyze global causes
    print("🌍 Analyzing global causes of death...")
    global_avg = analyze_global_causes(melted_df)
    plot_global_causes(global_avg)
    
    # Define causes of interest for deeper analysis
    causes_of_interest = [
        "Cardiovascular diseases",
        "Neoplasms",
        "Chronic respiratory diseases"
    ]
    
    # Analyze regional trends
    print("🗺️ Analyzing regional trends...")
    regional_data = analyze_regional_trends(recent_data, causes_of_interest)
    
    # Plot regional comparisons only if we have data
    regional_images_created = []
    for cause in causes_of_interest:
        if plot_regional_comparison(regional_data, cause):
            regional_images_created.append(f"regional_{cause.lower().replace(' ', '_')}.png")
    
    # Analyze temporal trends
    print("📈 Analyzing temporal trends...")
    countries_of_interest = ["United States", 'United Kingdom', "Japan", "Brazil", "India"]
    temporal_data = analyze_temporal_trends(df, causes_of_interest, countries_of_interest)
    
    # Plot temporal trends only if we have data
    trend_images_created = []
    for cause in causes_of_interest:
        if plot_temporal_trends(temporal_data, cause, countries_of_interest):
            trend_images_created.append(f"trend_{cause.lower().replace(' ', '_')}.png")
    
    # Generate statistics
    print("📋 Generating statistics...")
    stats = generate_statistics(df, global_avg)
    
    # Generate PDF report
    print("📄 Generating comprehensive PDF report...")
    generate_pdf_report(global_avg, regional_data, temporal_data, stats)
    
    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS".center(70))
    print("="*70)
    print(f"🏆 Leading cause of death: {stats['top_cause']} ({stats['top_cause_rate']:.1f} per 100k)")
    print(f"🥈 Second leading cause: {stats['second_cause']} ({stats['second_cause_rate']:.1f} per 100k)")
    print(f"📊 Based on sample data from {stats['total_countries']} countries")
    print("="*70)
    
    # Cleanup temporary files
    image_files = ["global_causes.png"] + regional_images_created + trend_images_created
    
    for file in image_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up: {file}")
    
    print("✅ Analysis complete! Report generated successfully.")

# -------------------------------
# Run the analysis
# -------------------------------
if __name__ == "__main__":
    main()