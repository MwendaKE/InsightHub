import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')

# -------------------------------
# Enhanced Data Loading Functions
# -------------------------------
def safe_csv_loader(file_path, expected_cols=None, skip_rows=4):
    """Safe CSV loader with error handling"""
    try:
        df = pd.read_csv(file_path, skiprows=skip_rows, engine='python')
        df.columns = [col.strip().replace('"', '') for col in df.columns]
        return df
    except:
        return pd.DataFrame()

def load_hiv_data(file_path):
    """Load and process HIV data"""
    df = safe_csv_loader(file_path, expected_cols=['Country Name'])
    if df.empty:
        return df

    year_columns = [col for col in df.columns if str(col).isdigit()]
    if year_columns:
        df = df[['Country Name'] + year_columns]
        df = df.melt(
            id_vars=['Country Name'],
            value_vars=year_columns,
            var_name='Year',
            value_name='Value'
        )
        df.rename(columns={'Country Name': 'Country'}, inplace=True)

    df.dropna(subset=['Value'], inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value', 'Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    
    # Scale if needed
    if df['Value'].max() > 100:
        df['Value'] = df['Value'] / 100
    
    return df

def load_population_data(file_path):
    """Load and process population data"""
    df = safe_csv_loader(file_path, expected_cols=['Country Name'])
    if df.empty:
        return df

    year_columns = [col for col in df.columns if str(col).isdigit()]
    if year_columns:
        df = df[['Country Name'] + year_columns]
        df = df.melt(
            id_vars=['Country Name'],
            value_vars=year_columns,
            var_name='Year',
            value_name='Population'
        )
        df.rename(columns={'Country Name': 'Country'}, inplace=True)

    df.dropna(subset=['Population'], inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    df.dropna(subset=['Population', 'Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    
    return df

# -------------------------------
# Enhanced Analysis Functions
# -------------------------------
def global_trend_analysis(df):
    """Analyze global HIV trends"""
    global_trend = df.groupby('Year')['Value'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(global_trend.index, global_trend.values, marker='o', linewidth=2.5, 
             markersize=6, color='#2E86AB', alpha=0.8)
    plt.fill_between(global_trend.index, global_trend.values, alpha=0.2, color='#2E86AB')
    
    plt.title('Global HIV Prevalence Trend (1990-2024)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('HIV Prevalence (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("global_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return global_trend

def top_bottom_countries(df, year, top_n=10):
    """Analyze top and bottom countries"""
    latest_data = df[df['Year'] == year].copy()
    
    top = latest_data.nlargest(top_n, 'Value')
    bottom = latest_data.nsmallest(top_n, 'Value')

    # Top countries chart
    plt.figure(figsize=(14, 8))
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, top_n))
    bars = plt.barh(top['Country'], top['Value'], color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold')
    
    plt.title(f'Top {top_n} Countries by HIV Prevalence ({year})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HIV Prevalence (%)', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("top_countries.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Bottom countries chart
    plt.figure(figsize=(14, 6))
    colors = plt.cm.Greens(np.linspace(0.6, 0.9, top_n))
    bars = plt.barh(bottom['Country'], bottom['Value'], color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}%', ha='left', va='center', fontweight='bold')
    
    plt.title(f'Countries with Lowest HIV Prevalence ({year})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HIV Prevalence (%)', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("bottom_countries.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return top, bottom

def regional_analysis(df):
    """Analyze regional patterns"""
    # Group by region-like countries (simplified)
    regions = ['Africa', 'Asia', 'Europe', 'America', 'Middle']
    regional_data = {}
    
    for region in regions:
        region_df = df[df['Country'].str.contains(region, case=False)]
        if not region_df.empty:
            regional_data[region] = region_df.groupby('Year')['Value'].mean()
    
    plt.figure(figsize=(12, 6))
    for region, data in regional_data.items():
        plt.plot(data.index, data.values, marker='o', linewidth=2, label=region, markersize=4)
    
    plt.title('HIV Prevalence by Region', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('HIV Prevalence (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regional_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return regional_data

def outlier_analysis(df):
    """Analyze countries with significant changes"""
    earliest_year = df['Year'].min()
    latest_year = df['Year'].max()
    
    change_df = df.pivot_table(index='Country', columns='Year', values='Value')
    change_df = change_df.dropna(how='all')
    
    # Calculate percentage change
    change_df['Change'] = ((change_df[latest_year] - change_df[earliest_year]) / change_df[earliest_year]) * 100
    change_df['Absolute_Change'] = change_df[latest_year] - change_df[earliest_year]
    
    # Get top increases and decreases
    increase = change_df.nlargest(5, 'Absolute_Change')
    decrease = change_df.nsmallest(5, 'Absolute_Change')
    
    return increase, decrease

def hiv_absolute_numbers(df_hiv, df_pop):
    """Calculate absolute HIV cases"""
    df_combined = pd.merge(df_hiv, df_pop, on=['Country', 'Year'], how='inner')
    df_combined['HIV_Cases'] = (df_combined['Value'] / 100) * df_combined['Population']
    return df_combined

def generate_statistics(df_hiv, df_pop, df_absolute):
    """Generate comprehensive statistics"""
    stats = {
        'total_countries': df_hiv['Country'].nunique(),
        'years_covered': f"{df_hiv['Year'].min()} - {df_hiv['Year'].max()}",
        'global_prevalence_current': df_hiv[df_hiv['Year'] == df_hiv['Year'].max()]['Value'].mean(),
        'global_prevalence_peak': df_hiv.groupby('Year')['Value'].mean().max(),
        'peak_year': df_hiv.groupby('Year')['Value'].mean().idxmax(),
        'total_cases_current': df_absolute[df_absolute['Year'] == df_absolute['Year'].max()]['HIV_Cases'].sum()
    }
    return stats

# -------------------------------
# Enhanced PDF Report Generation
# -------------------------------
def generate_pdf(global_trend, top10, bottom10, increase, decrease, stats, filename="HIV_Analysis_Report.pdf"):
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
    
    # Title Page
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-100, "HIV PREVALENCE ANALYSIS REPORT")
    
    c.setFillColor(secondary_color)
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-150, "Comprehensive Global HIV Trends Analysis")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-200, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-250, f"Analysed by Mwenda E. Njagi @ Github.com/MwendaKE/InsightHub")
    
    c.setFillColor(HexColor('#666666'))
    c.setFont("Helvetica", 11)
    c.drawCentredString(width/2, height-300, "Data Source: World Development Indicators")
    
    # Add some decorative elements
    c.setStrokeColor(primary_color)
    c.setLineWidth(1)
    c.line(50, height-280, width-50, height-280)
    
    add_footer()  # Add footer to first page
    c.showPage()  # End of page 1
    
    # Executive Summary
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-50, "Executive Summary")
    
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica", 10)
    summary_text = [
        f"• Analyzed HIV prevalence data from {stats['total_countries']} countries",
        f"• Coverage period: {stats['years_covered']}",
        f"• Current global prevalence: {stats['global_prevalence_current']:.3f}%",
        f"• Peak prevalence: {stats['global_prevalence_peak']:.3f}% in {stats['peak_year']}",
        f"• Estimated total cases: {stats['total_cases_current']:,.0f} people",
        "",
        "Key Findings:",
        "• Southern African nations show disproportionately high prevalence rates",
        "• Global trends indicate stabilization after peak years",
        "• Significant progress in treatment access and prevention"
    ]
    
    y_pos = height-80
    for line in summary_text:
        c.drawString(70, y_pos, line)
        y_pos -= 20
    
    # Global Trend
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_pos-40, "Global HIV Prevalence Trend")
    c.drawImage("global_trend.png", 50, y_pos-270, width=500, height=200)
    
    # Analysis Text
    analysis_text = [
        "The global HIV prevalence trend shows a clear pattern:",
        "• Rapid increase from 1990s to early 2000s",
        "• Peak around 2004-2006 due to improved detection and reporting",
        "• Gradual decline post-2010, reflecting successful intervention programs",
        "• Current stabilization suggests effective management strategies",
        "",
        "This trend reflects the success of global health initiatives, improved",
        "antiretroviral therapy access, and better prevention education."
    ]
    
    y_text = y_pos - 290
    for line in analysis_text:
        c.drawString(70, y_text, line)
        y_text -= 15
    
    add_footer()  # Add footer to second page
    c.showPage()  # End of page 2
    
    # Top Countries - Page 3
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Top 10 Countries by HIV Prevalence")
    c.drawImage("top_countries.png", 50, height-250, width=500, height=180)
    
    # Top Countries Analysis
    top_analysis = [
        "High prevalence countries share common characteristics:",
        "• Limited healthcare infrastructure in rural areas",
        "• Economic challenges affecting prevention programs",
        "• Cultural factors and stigma around testing",
        "• Historical patterns of disease transmission",
        "",
        "Countries like Eswatini, Lesotho, and Botswana show:",
        "• Prevalence rates above 20%, indicating severe epidemics",
        "• Need for targeted international support",
        "• Success stories in some regions show progress is possible"
    ]
    
    y_top = height-450
    for line in top_analysis:
        c.drawString(70, y_top, line)
        y_top -= 15
    
    add_footer()  # Add footer to third page
    c.showPage()  # End of page 3
    
    # Bottom Countries - Page 4
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Countries with Lowest HIV Prevalence")
    c.drawImage("bottom_countries.png", 50, height-250, width=500, height=180)
    
    # Bottom Countries Analysis
    bottom_analysis = [
        "Low prevalence countries demonstrate successful strategies:",
        "• Comprehensive sex education programs",
        "• Widespread availability of condoms and prevention tools",
        "• Strong healthcare systems and early detection",
        "• Cultural openness about sexual health",
        "",
        "Key success factors include:",
        "• Government commitment to HIV prevention",
        "• International cooperation and funding",
        "• Community-based education programs",
        "• Integration of HIV services with general healthcare"
    ]
    
    y_bottom = height-450
    for line in bottom_analysis:
        c.drawString(70, y_bottom, line)
        y_bottom -= 15
    
    add_footer()  # Add footer to fourth page
    c.showPage()  # End of page 4
    
    # Significant Changes - Page 5
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Notable Changes in HIV Prevalence")
    
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, height-80, "Largest Increases:")
    
    c.setFont("Helvetica", 10)
    y_increase = height-100
    for i, (country, row) in enumerate(increase.iterrows()):
        c.drawString(90, y_increase, f"{country}: +{row['Absolute_Change']:.2f}% ({row['Change']:.1f}% change)")
        y_increase -= 15
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, y_increase-10, "Largest Decreases:")
    
    c.setFont("Helvetica", 10)
    y_decrease = y_increase - 25
    for i, (country, row) in enumerate(decrease.iterrows()):
        c.drawString(90, y_decrease, f"{country}: {row['Absolute_Change']:.2f}% ({row['Change']:.1f}% change)")
        y_decrease -= 15
    
    # Change Analysis
    change_analysis = [
        "",
        "Reasons for significant changes:",
        "INCREASES may be due to:",
        "• Improved testing and case detection",
        "• Population growth in affected areas",
        "• Breakdown of healthcare systems",
        "• Emergence of drug-resistant strains",
        "",
        "DECREASES typically result from:",
        "• Successful prevention programs",
        "• Widespread antiretroviral therapy",
        "• Behavioral changes and education",
        "• International aid and support"
    ]
    
    y_change = y_decrease - 10
    for line in change_analysis:
        c.drawString(70, y_change, line)
        y_change -= 15
    
    add_footer()  # Add footer to fifth page
    c.showPage()  # End of page 5
    
    # Recommendations - Page 6
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Recommendations and Future Directions")
    
    recommendations = [
        "1. TARGETED INTERVENTIONS:",
        "   • Focus resources on high-prevalence regions",
        "   • Customize programs to local cultural contexts",
        "   • Address economic barriers to healthcare access",
        "",
        "2. PREVENTION STRATEGIES:",
        "   • Expand comprehensive sex education",
        "   • Increase availability of prevention tools",
        "   • Combat stigma through public awareness",
        "",
        "3. TREATMENT ACCESS:",
        "   • Improve antiretroviral therapy availability",
        "   • Strengthen healthcare infrastructure",
        "   • Support research for better treatments",
        "",
        "4. GLOBAL COOPERATION:",
        "   • Maintain international funding commitments",
        "   • Share successful strategies across borders",
        "   • Coordinate research and development efforts",
        "",
        "5. DATA-DRIVEN APPROACH:",
        "   • Continue robust surveillance and reporting",
        "   • Use data to identify emerging trends",
        "   • Evaluate program effectiveness regularly"
    ]
    
    y_rec = height-80
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica", 10)
    for line in recommendations:
        c.drawString(70, y_rec, line)
        y_rec -= 15
        if y_rec < 50:  # Handle text overflow
            add_footer()
            c.showPage()
            y_rec = height-50
            c.setFont("Helvetica", 10)
            add_footer()
    
    add_footer()  # Add footer to final page
    
    c.save()
    print(f"✅ Comprehensive PDF report generated: {filename}")
       
# -------------------------------
# Main Function
# -------------------------------
def main():
    print("🚀 Starting HIV Data Analysis...")
    
    # Load data
    print("📊 Loading data...")
    hiv_df = load_hiv_data("../Data Sets/hiv_prevalence.csv")
    pop_df = load_population_data("../Data Sets/population.csv")
    
    if hiv_df.empty or pop_df.empty:
        print("❌ Error: Could not load data files")
        return
    
    print(f"✅ HIV data loaded: {hiv_df.shape[0]} records, {hiv_df['Country'].nunique()} countries")
    print(f"✅ Population data loaded: {pop_df.shape[0]} records")
    
    # Analysis
    print("📈 Analyzing global trends...")
    global_trend = global_trend_analysis(hiv_df)
    
    latest_year = hiv_df['Year'].max()
    print(f"📅 Latest year in data: {latest_year}")
    
    print("🏆 Analyzing top and bottom countries...")
    top10, bottom10 = top_bottom_countries(hiv_df, latest_year)
    
    print("📊 Analyzing regional patterns...")
    regional_analysis(hiv_df)
    
    print("🔍 Identifying significant changes...")
    increase, decrease = outlier_analysis(hiv_df)
    
    print("🧮 Calculating absolute cases...")
    df_absolute = hiv_absolute_numbers(hiv_df, pop_df)
    
    print("📋 Generating statistics...")
    stats = generate_statistics(hiv_df, pop_df, df_absolute)
    
    # Generate PDF report
    print("📄 Generating comprehensive PDF report...")
    generate_pdf(global_trend, top10, bottom10, increase, decrease, stats)
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS".center(60))
    print("="*60)
    print(f"🌍 Global prevalence: {stats['global_prevalence_current']:.3f}%")
    print(f"📈 Peak was {stats['global_prevalence_peak']:.3f}% in {stats['peak_year']}")
    print(f"👥 Estimated total cases: {stats['total_cases_current']:,.0f} people")
    print(f"🏆 Top country: {top10.iloc[0]['Country']} ({top10.iloc[0]['Value']:.2f}%)")
    print(f"📉 Largest increase: {increase.index[0]} (+{increase.iloc[0]['Absolute_Change']:.2f}%)")
    print("="*60)
    
    # Cleanup temporary files
    for file in ["global_trend.png", "top_countries.png", "bottom_countries.png", "regional_trends.png"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("✅ Analysis complete! Report generated successfully.")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()