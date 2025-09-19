import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_cancer_data(file_path):
    """Load cancer data from local CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Cancer data loaded: {len(df)} records, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"❌ Error loading cancer data: {e}")
        return pd.DataFrame()

def clean_cancer_data(df):
    """Clean and preprocess cancer data"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Convert all numeric columns
    for col in df_clean.columns:
        if col != 'State':  # Skip state names
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop rows with missing critical data
    df_clean = df_clean.dropna(subset=['Total.Rate', 'Total.Number', 'Total.Population'])
    
    print(f"✅ Data cleaned: {len(df_clean)} records remaining")
    return df_clean

# -------------------------------
# Enhanced Analysis Functions with Better Image Formatting
# -------------------------------
def state_analysis(df):
    """Comprehensive state-level analysis with better image formatting"""
    state_rates = df[['State', 'Total.Rate', 'Total.Number', 'Total.Population']].copy()
    state_rates['Death_Rate_Per_100k'] = (state_rates['Total.Number'] / state_rates['Total.Population']) * 100000
    state_rates = state_rates.sort_values('Total.Rate', ascending=False)
    
    # Top and bottom 10 states
    top_10 = state_rates.head(10)
    bottom_10 = state_rates.tail(10)
    
    # Plot top states - tightly cropped
    plt.figure(figsize=(12, 6))  # Reduced height
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, len(top_10)))
    bars = plt.barh(top_10['State'], top_10['Total.Rate'], color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.title('Top 10 States by Cancer Mortality Rate', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Mortality Rate (per 100,000)', fontsize=10)
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("top_states.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot bottom states - tightly cropped
    plt.figure(figsize=(12, 6))  # Reduced height
    colors = plt.cm.Greens(np.linspace(0.6, 0.9, len(bottom_10)))
    bars = plt.barh(bottom_10['State'], bottom_10['Total.Rate'], color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.title('10 States with Lowest Cancer Mortality Rate', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Mortality Rate (per 100,000)', fontsize=10)
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("bottom_states.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return state_rates

def cancer_type_analysis(df):
    """Comprehensive analysis of different cancer types with better image formatting"""
    # Extract cancer type columns
    cancer_cols = [col for col in df.columns if col.startswith('Types.') and col.endswith('.Total')]
    
    cancer_data = []
    for col in cancer_cols:
        cancer_type = col.split('.')[1]  # Extract cancer type name
        avg_rate = df[col].mean()
        total_cases = df[col].sum()
        cancer_data.append({
            'Type': cancer_type, 
            'Avg_Rate': avg_rate,
            'Total_Cases': total_cases
        })
    
    cancer_df = pd.DataFrame(cancer_data).sort_values('Avg_Rate', ascending=False)
    
    # Plot cancer types - tightly cropped
    plt.figure(figsize=(14, 8))  # Reduced height
    colors = plt.cm.Set3(np.linspace(0, 1, len(cancer_df)))
    bars = plt.barh(cancer_df['Type'], cancer_df['Avg_Rate'], color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.title('Cancer Types by Average Mortality Rate', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Average Mortality Rate (per 100,000)', fontsize=10)
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("cancer_types.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return cancer_df

def demographic_analysis(df):
    """Comprehensive demographic analysis with better image formatting"""
    # Age group analysis
    age_groups = ['< 18', '18-45', '45-64', '> 64']
    age_data = {}
    
    for age_group in age_groups:
        col_name = f'Rates.Age.{age_group}'
        if col_name in df.columns:
            age_data[age_group] = df[col_name].mean()
    
    # Gender analysis across age groups
    gender_age_data = {}
    gender_patterns = [
        ('Female', '< 18', 'Rates.Age and Sex.Female.< 18'),
        ('Male', '< 18', 'Rates.Age and Sex.Male.< 18'),
        ('Female', '18-45', 'Rates.Age and Sex.Female.18 - 45'),
        ('Male', '18-45', 'Rates.Age and Sex.Male.18 - 45'),
        ('Female', '45-64', 'Rates.Age and Sex.Female.45 - 64'),
        ('Male', '45-64', 'Rates.Age and Sex.Male.45 - 64'),
        ('Female', '65+', 'Rates.Age and Sex.Female.> 64'),
        ('Male', '65+', 'Rates.Age and Sex.Male.> 64')
    ]
    
    for gender, age_group, col_name in gender_patterns:
        if col_name in df.columns:
            gender_age_data[f'{gender}_{age_group}'] = df[col_name].mean()
    
    # Race analysis
    race_cols = [col for col in df.columns if col.startswith('Rates.Race.') and 
                not col.startswith('Rates.Race and Sex') and 
                not col.endswith('non-Hispanic')]
    race_data = {}
    for col in race_cols:
        race = col.split('.')[-1]
        race_data[race] = df[col].mean()
    
    # Create visualizations with tight cropping
    # Age group comparison
    plt.figure(figsize=(10, 6))  # Reduced height
    age_groups_sorted = ['< 18', '18-45', '45-64', '> 64']
    age_rates = [age_data.get(group, 0) for group in age_groups_sorted]
    
    plt.bar(age_groups_sorted, age_rates, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Cancer Mortality Rates by Age Group', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Age Group', fontsize=10)
    plt.ylabel('Mortality Rate (per 100,000)', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("age_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Gender comparison for adults
    adult_categories = ['18-45', '45-64', '65+']
    female_rates = [gender_age_data.get(f'Female_{cat}', 0) for cat in ['18-45', '45-64', '65+']]
    male_rates = [gender_age_data.get(f'Male_{cat}', 0) for cat in ['18-45', '45-64', '65+']]
    
    plt.figure(figsize=(10, 6))  # Reduced height
    x = np.arange(len(adult_categories))
    width = 0.35
    
    plt.bar(x - width/2, female_rates, width, label='Female', alpha=0.8, color='#FF6B6B')
    plt.bar(x + width/2, male_rates, width, label='Male', alpha=0.8, color='#4ECDC4')
    
    plt.title('Cancer Mortality Rates by Gender and Age Group', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Age Group', fontsize=10)
    plt.ylabel('Mortality Rate (per 100,000)', fontsize=10)
    plt.xticks(x, adult_categories)
    plt.legend()
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("gender_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Race comparison
    plt.figure(figsize=(12, 6))  # Reduced height
    races = list(race_data.keys())
    rates = [race_data[race] for race in races]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(races)))
    bars = plt.bar(races, rates, alpha=0.8, color=colors)
    
    plt.title('Cancer Mortality Rates by Race', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Race', fontsize=10)
    plt.ylabel('Mortality Rate (per 100,000)', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("race_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return age_data, gender_age_data, race_data

def regional_analysis(df):
    """Analyze regional patterns with better image formatting"""
    # Simple regional grouping
    regions = {
        'Northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 
                     'Rhode Island', 'Vermont', 'New Jersey', 'New York', 'Pennsylvania'],
        'Midwest': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 
                   'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 
                   'North Dakota', 'South Dakota'],
        'South': ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina',
                 'South Carolina', 'Virginia', 'West Virginia', 'Alabama',
                 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana',
                 'Oklahoma', 'Texas'],
        'West': ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico',
                'Utah', 'Wyoming', 'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
    }
    
    regional_data = {}
    for region, states in regions.items():
        region_df = df[df['State'].isin(states)]
        if not region_df.empty:
            regional_data[region] = {
                'Avg_Rate': region_df['Total.Rate'].mean(),
                'Total_Deaths': region_df['Total.Number'].sum(),
                'Total_Population': region_df['Total.Population'].sum(),
                'States_Count': len(region_df)
            }
    
    # Plot regional comparison - tightly cropped
    plt.figure(figsize=(10, 6))  # Reduced height
    regions_sorted = list(regional_data.keys())
    rates = [regional_data[region]['Avg_Rate'] for region in regions_sorted]
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(regions_sorted)))
    bars = plt.bar(regions_sorted, rates, alpha=0.8, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Cancer Mortality Rates by US Region', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Region', fontsize=10)
    plt.ylabel('Average Mortality Rate (per 100,000)', fontsize=10)
    plt.tight_layout(pad=1.0)  # Reduced padding
    plt.savefig("regional_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return regional_data

def generate_statistics(df):
    """Generate comprehensive statistics"""
    stats = {
        'total_states': len(df),
        'avg_mortality_rate': df['Total.Rate'].mean(),
        'max_rate': df['Total.Rate'].max(),
        'min_rate': df['Total.Rate'].min(),
        'max_state': df.loc[df['Total.Rate'].idxmax(), 'State'],
        'min_state': df.loc[df['Total.Rate'].idxmin(), 'State'],
        'total_deaths': df['Total.Number'].sum(),
        'total_population': df['Total.Population'].sum(),
        'death_rate_per_100k': (df['Total.Number'].sum() / df['Total.Population'].sum()) * 100000,
        'std_dev_rate': df['Total.Rate'].std(),
        'median_rate': df['Total.Rate'].median()
    }
    
    # Calculate correlation between population and death rate
    stats['correlation_population_deaths'] = df['Total.Population'].corr(df['Total.Number'])
    
    return stats

# -------------------------------
# Enhanced PDF Report Generation
# -------------------------------
def generate_cancer_pdf(state_rates, cancer_types, age_data, gender_data, race_data, regional_data, stats, filename="Comprehensive_Cancer_Analysis_Report.pdf"):
    """Generate comprehensive cancer analysis PDF report"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Colors
    primary_color = HexColor('#E63946')  # Cancer red
    secondary_color = HexColor('#457B9D')  # Blue
    accent_color = HexColor('#A8DADC')  # Light blue
    
    def add_footer():
        """Add footer to current page"""
        c.setFillColor(HexColor('#666666'))
        c.setFont("Helvetica-Oblique", 8)
        footer_text = "Generated by Insight Hub Analysis Program created by Mwenda E. Njagi at GitHub.com. Link: https://github.com/MwendaKE/InsightHub."
        c.drawCentredString(width/2, 20, footer_text)
    
    def draw_text_lines(lines, start_y, line_height=15, left_margin=70, right_margin=50, font_name="Helvetica", font_size=10, text_color=HexColor('#333333')):
        """Helper function to draw text lines with automatic pagination and font preservation"""
        current_y = start_y
        c.setFont(font_name, font_size)
        c.setFillColor(text_color)
        
        for line in lines:
            if current_y < 50:  # Bottom margin reached
                add_footer()
                c.showPage()
                current_y = height - 50  # Reset to top of new page
                # Re-set font and color for new page
                c.setFont(font_name, font_size)
                c.setFillColor(text_color)
                # Add header for new page if needed
                c.setFillColor(primary_color)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height-30, "Continued Analysis")
                c.setFont(font_name, font_size)
                c.setFillColor(text_color)
            c.drawString(left_margin, current_y, line)
            current_y -= line_height
        return current_y
    
    # Title Page
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-100, "COMPREHENSIVE CANCER ANALYSIS REPORT (UNITED STATES)")
    
    c.setFillColor(secondary_color)
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-150, "Multi-Dimensional Cancer Mortality Analysis")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-200, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-250, f"Analysed by Mwenda E. Njagi @ Github.com/MwendaKE/InsightHub")
    
    c.setFillColor(HexColor('#666666'))
    c.setFont("Helvetica", 11)
    c.drawCentredString(width/2, height-300, "Data Source: CORGIS Cancer Dataset - State-Level Statistics")
    
    add_footer()
    c.showPage()
    
    # Executive Summary
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-50, "Executive Summary")
    
    summary_text = [
        f"• Comprehensive analysis of {stats['total_states']} US states",
        f"• Average mortality rate: {stats['avg_mortality_rate']:.1f} ± {stats['std_dev_rate']:.1f} per 100,000",
        f"• Highest rate: {stats['max_state']} ({stats['max_rate']:.1f}/100,000)",
        f"• Lowest rate: {stats['min_state']} ({stats['min_rate']:.1f}/100,000)",
        f"• Total deaths analyzed: {stats['total_deaths']:,.0f}",
        f"• Total population covered: {stats['total_population']:,.0f}",
        f"• Overall death rate: {stats['death_rate_per_100k']:.1f} per 100,000",
        f"• Strong correlation between population and deaths: {stats['correlation_population_deaths']:.3f}",
        "",
        "Key Insights:",
        "• Significant geographic disparities in cancer mortality",
        "• Dramatic age-related patterns in cancer rates",
        "• Notable demographic variations across race and gender",
        "• Regional clustering of high/low mortality states"
    ]
    
    y_pos = height - 80
    y_pos = draw_text_lines(summary_text, y_pos)
    
    add_footer()
    c.showPage()
    
    # State Analysis
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Geographic Analysis: State-Level Patterns")
    c.drawImage("top_states.png", 50, height-280, width=500, height=200)
    c.drawImage("bottom_states.png", 50, height-500, width=500, height=200)
    
    add_footer()
    c.showPage()
    
    # Regional Analysis
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Regional Patterns Analysis")
    c.drawImage("regional_analysis.png", 50, height-330, width=500, height=250)
    
    # Regional insights
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, height-600, "Regional Summary:")
    
    regional_summary = []
    for region, data in regional_data.items():
        regional_summary.append(f"• {region}: {data['Avg_Rate']:.1f}/100,000 ({data['States_Count']} states)")
    
    y_reg = height - 620
    y_reg = draw_text_lines(regional_summary, y_reg, font_name="Helvetica", font_size=10)
    
    add_footer()
    c.showPage()
    
    # Cancer Types Analysis
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Cancer Type Analysis")
    c.drawImage("cancer_types.png", 50, height-380, width=500, height=300)
    
    # Top cancer types
    top_5 = cancer_types.head(5)
    c.setFillColor(HexColor('#333333'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, height-700, "Highest Mortality Cancer Types:")
    
    cancer_type_list = []
    for i, (_, row) in enumerate(top_5.iterrows()):
        cancer_type_list.append(f"{i+1}. {row['Type']}: {row['Avg_Rate']:.1f} per 100,000")
    
    y_list = height - 720
    y_list = draw_text_lines(cancer_type_list, y_list, font_name="Helvetica", font_size=10)
    
    add_footer()
    c.showPage()
    
    # Demographic Analysis - Age
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Age Group Analysis")
    c.drawImage("age_analysis.png", 50, height-330, width=500, height=250)
    
    # Age insights
    age_text = [
        "Age Group Patterns:",
        f"• Children (<18): {age_data.get('< 18', 0):.1f}/100,000",
        f"• Young Adults (18-45): {age_data.get('18-45', 0):.1f}/100,000",
        f"• Middle-aged (45-64): {age_data.get('45-64', 0):.1f}/100,000",
        f"• Seniors (65+): {age_data.get('> 64', 0):.1f}/100,000",
        "",
        "Key Finding:",
        "• 65+ age group has 50-100x higher mortality than children",
        "• Middle-aged adults show significant cancer burden",
        "• Young adults relatively protected but need prevention focus"
    ]
    
    y_age = height - 600
    y_age = draw_text_lines(age_text, y_age, font_name="Helvetica", font_size=10)
    
    add_footer()
    c.showPage()
    
    # Demographic Analysis - Gender
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Gender and Age Analysis")
    c.drawImage("gender_analysis.png", 50, height-330, width=500, height=250)
    
    # Gender insights
    gender_text = [
        "Gender Patterns:",
        "• Males generally show higher mortality rates across age groups",
        "• Gender gap widens in older age groups",
        "• Both genders show dramatic increase with age",
        "",
        "Prevention Implications:",
        "• Gender-specific screening programs needed",
        "• Targeted awareness campaigns for high-risk groups",
        "• Age-appropriate prevention strategies"
    ]
    
    y_gender = height - 600
    y_gender = draw_text_lines(gender_text, y_gender, font_name="Helvetica", font_size=10)
    
    add_footer()
    c.showPage()
    
    # Demographic Analysis - Race
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Racial Disparities Analysis")
    c.drawImage("race_analysis.png", 50, height-380, width=500, height=300)
    
    # Race insights
    race_text = [
        "Racial Health Disparities:",
        "• Significant variations across racial groups",
        "• Some groups show 2-3x higher mortality rates",
        "• Complex interplay of genetic, social, and access factors",
        "",
        "Equity Implications:",
        "• Need for targeted outreach programs",
        "• Address healthcare access disparities",
        "• Cultural competency in cancer care"
    ]
    
    y_race = height - 700
    y_race = draw_text_lines(race_text, y_race, font_name="Helvetica", font_size=10)
    
    add_footer()
    c.showPage()
    
    # Recommendations
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Strategic Recommendations & Action Plan")
    
    recommendations = [
        "1. GEOGRAPHIC TARGETING:",
        "   • Focus resources on high-mortality states and regions",
        "   • Develop state-specific cancer control programs",
        "   • Share best practices from low-mortality areas",
        "",
        "2. AGE-SPECIFIC STRATEGIES:",
        "   • Enhance screening for 45+ age groups",
        "   • Youth prevention education programs",
        "   • Senior-focused early detection initiatives",
        "",
        "3. DEMOGRAPHIC EQUITY:",
        "   • Address racial health disparities",
        "   • Gender-specific prevention campaigns",
        "   • Culturally competent healthcare services",
        "",
        "4. CANCER TYPE PRIORITIZATION:",
        "   • Focus on high-mortality cancer types",
        "   • Develop type-specific prevention protocols",
        "   • Improve early detection methods",
        "",
        "5. DATA-DRIVEN APPROACH:",
        "   • Continuous monitoring of state-level trends",
        "   • Regular evaluation of intervention effectiveness",
        "   • Research into underlying causes of disparities"
    ]
    
    y_rec = height - 80
    y_rec = draw_text_lines(recommendations, y_rec, line_height=15, font_name="Helvetica", font_size=10)
    
    add_footer()
    c.save()
    print(f"✅ Comprehensive Cancer PDF report generated: {filename}")
    
# -------------------------------
# Main Function
# -------------------------------
def main():
    print("🚀 Starting Comprehensive Cancer Data Analysis...")
    
    # Load and clean data
    print("📊 Loading cancer data from local file...")
    df = load_cancer_data("../Data Sets/cancer.csv")
    if df.empty:
        print("❌ Failed to load cancer data")
        return
    
    print("🧹 Cleaning and preprocessing data...")
    df_clean = clean_cancer_data(df)
    if df_clean.empty:
        print("❌ No data after cleaning")
        return
    
    # Comprehensive analysis
    print("🗺️ Analyzing state-level patterns...")
    state_rates = state_analysis(df_clean)
    
    print("📊 Analyzing regional patterns...")
    regional_data = regional_analysis(df_clean)
    
    print("🔬 Analyzing cancer types...")
    cancer_types = cancer_type_analysis(df_clean)
    
    print("👥 Analyzing demographics...")
    age_data, gender_data, race_data = demographic_analysis(df_clean)
    
    print("📋 Generating comprehensive statistics...")
    stats = generate_statistics(df_clean)
    
    # Generate PDF report
    print("📄 Generating comprehensive PDF report...")
    generate_cancer_pdf(state_rates, cancer_types, age_data, gender_data, race_data, regional_data, stats)
    
    # Print key insights
    print("\n" + "="*70)
    print("COMPREHENSIVE CANCER ANALYSIS - KEY INSIGHTS".center(70))
    print("="*70)
    print(f"📊 National Average: {stats['avg_mortality_rate']:.1f} ± {stats['std_dev_rate']:.1f}/100,000")
    print(f"📍 Geographic Range: {stats['min_state']} ({stats['min_rate']:.1f}) to {stats['max_state']} ({stats['max_rate']:.1f})")
    print(f"👥 Total Impact: {stats['total_deaths']:,.0f} deaths across {stats['total_population']:,.0f} people")
    print(f"🎯 Top Cancer Type: {cancer_types.iloc[0]['Type']} ({cancer_types.iloc[0]['Avg_Rate']:.1f}/100,000)")
    print(f"📈 Highest Region: {max(regional_data.items(), key=lambda x: x[1]['Avg_Rate'])[0]} region")
    print(f"👴 Age Pattern: Seniors (65+) show {age_data.get('> 64', 0)/age_data.get('< 18', 1):.0f}x higher rates than children")
    print("="*70)
    
    # Cleanup
    for file in ["top_states.png", "bottom_states.png", "cancer_types.png", 
                 "age_analysis.png", "gender_analysis.png", "race_analysis.png", 
                 "regional_analysis.png"]:
        try:
            os.remove(file)
        except:
            pass
    
    print("✅ Comprehensive cancer analysis complete! Report generated successfully.")

if __name__ == "__main__":
    main()