import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_sickle_cell_data(file_path):
    """Load sickle cell data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Sickle cell data loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"📅 Date range: {df['year'].min()} - {df['year'].max()}")
        print(f"🌍 Countries: {df['country'].nunique()}")
        return df
    except Exception as e:
        print(f"❌ Error loading sickle cell data: {e}")
        return pd.DataFrame()

def clean_sickle_cell_data(df):
    """Clean and preprocess sickle cell data"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Ensure numeric columns are properly formatted
    numeric_cols = ['deaths', 'prevalence', 'death_rate_per_100k', 'health_expenditure_pct_gdp', 
                   'gdp_per_capita_usd', 'life_expectancy', 'mortality_burden_score', 'healthcare_gap']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop rows with missing critical data
    df_clean = df_clean.dropna(subset=['deaths', 'death_rate_per_100k', 'country', 'year'])
    
    print(f"✅ Data cleaned: {len(df_clean)} records remaining")
    return df_clean

# -------------------------------
# Enhanced Visualization Functions
# -------------------------------
def create_global_trends_chart(df):
    """Create comprehensive global trends visualization"""
    yearly_trends = df.groupby('year').agg({
        'deaths': 'sum',
        'prevalence': 'sum',
        'death_rate_per_100k': 'mean',
        'gdp_per_capita_usd': 'mean',
        'life_expectancy': 'mean'
    }).reset_index()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Deaths over time
    ax1.plot(yearly_trends['year'], yearly_trends['deaths'], marker='o', linewidth=3, 
             color='#E63946', markersize=6)
    ax1.fill_between(yearly_trends['year'], yearly_trends['deaths'], alpha=0.3, color='#E63946')
    ax1.set_title('Global Sickle Cell Deaths Over Time (1990-2022)', fontweight='bold', fontsize=12, pad=10)
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Total Deaths', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Death rate over time
    ax2.plot(yearly_trends['year'], yearly_trends['death_rate_per_100k'], marker='s', linewidth=3, 
             color='#457B9D', markersize=6)
    ax2.fill_between(yearly_trends['year'], yearly_trends['death_rate_per_100k'], alpha=0.3, color='#457B9D')
    ax2.set_title('Average Death Rate Trend (1990-2022)', fontweight='bold', fontsize=12, pad=10)
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Death Rate (per 100,000)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Life expectancy vs GDP
    scatter = ax3.scatter(df['gdp_per_capita_usd'], df['life_expectancy'], 
                         c=df['death_rate_per_100k'], cmap='Reds', alpha=0.7, s=50)
    ax3.set_title('Economic Development vs Health Outcomes', fontweight='bold', fontsize=12, pad=10)
    ax3.set_xlabel('GDP per Capita (USD)', fontweight='bold')
    ax3.set_ylabel('Life Expectancy', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Death Rate')
    
    # Healthcare expenditure vs death rate
    ax4.scatter(df['health_expenditure_pct_gdp'], df['death_rate_per_100k'], 
               alpha=0.7, s=50, color='#E76F51')
    ax4.set_title('Healthcare Spending vs Mortality Rate', fontweight='bold', fontsize=12, pad=10)
    ax4.set_xlabel('Health Expenditure (% of GDP)', fontweight='bold')
    ax4.set_ylabel('Death Rate (per 100,000)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['health_expenditure_pct_gdp'], df['death_rate_per_100k'], 1)
    p = np.poly1d(z)
    ax4.plot(df['health_expenditure_pct_gdp'], p(df['health_expenditure_pct_gdp']), 
             "r--", alpha=0.8)
    
    plt.tight_layout(pad=3.0)
    plt.savefig("global_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return yearly_trends

def create_country_comparison_charts(df):
    """Create country-level comparison charts"""
    country_stats = df.groupby('country').agg({
        'deaths': 'mean',
        'death_rate_per_100k': 'mean',
        'prevalence': 'mean',
        'gdp_per_capita_usd': 'mean',
        'life_expectancy': 'mean',
        'health_expenditure_pct_gdp': 'mean'
    }).reset_index()
    
    # Top 10 countries by death rate
    top_countries = country_stats.nlargest(10, 'death_rate_per_100k')
    bottom_countries = country_stats.nsmallest(10, 'death_rate_per_100k')
    
    # Chart 1: Top countries by death rate
    plt.figure(figsize=(14, 8))
    colors = plt.cm.Reds(np.linspace(0.6, 1, len(top_countries)))
    bars = plt.barh(top_countries['country'], top_countries['death_rate_per_100k'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.title('Top 10 Countries by Sickle Cell Death Rate\n(1990-2022 Average)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Death Rate (per 100,000)', fontweight='bold', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("top_countries.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Death rate vs GDP scatter by region
    plt.figure(figsize=(12, 8))
    regions = df['who_region'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))
    
    for i, region in enumerate(regions):
        region_data = country_stats[country_stats['country'].isin(
            df[df['who_region'] == region]['country'].unique())]
        plt.scatter(region_data['gdp_per_capita_usd'], region_data['death_rate_per_100k'],
                   c=[colors[i]], label=region, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.title('Economic Development vs Sickle Cell Mortality by Region', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('GDP per Capita (USD)', fontweight='bold', fontsize=12)
    plt.ylabel('Death Rate (per 100,000)', fontweight='bold', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig("economic_vs_mortality.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return country_stats, top_countries, bottom_countries

def create_regional_analysis_charts(df):
    """Create regional analysis charts"""
    regional_stats = df.groupby('who_region').agg({
        'deaths': 'mean',
        'death_rate_per_100k': 'mean',
        'prevalence': 'mean',
        'gdp_per_capita_usd': 'mean',
        'life_expectancy': 'mean'
    }).reset_index()
    
    income_stats = df.groupby('income_level').agg({
        'deaths': 'mean',
        'death_rate_per_100k': 'mean',
        'gdp_per_capita_usd': 'mean',
        'life_expectancy': 'mean'
    }).reset_index()
    
    # Regional comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Regional death rates
    regions_sorted = regional_stats.sort_values('death_rate_per_100k', ascending=False)
    bars1 = ax1.bar(regions_sorted['who_region'], regions_sorted['death_rate_per_100k'], 
                   color=['#E63946', '#F4A261', '#2A9D8F', '#457B9D'], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Sickle Cell Death Rates by WHO Region', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel('Death Rate (per 100,000)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Income level death rates
    income_sorted = income_stats.sort_values('death_rate_per_100k', ascending=False)
    bars2 = ax2.bar(income_sorted['income_level'], income_sorted['death_rate_per_100k'],
                   color=['#E63946', '#F4A261', '#2A9D8F'], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Sickle Cell Death Rates by Income Level', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('Death Rate (per 100,000)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("regional_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Regional trend over time
    plt.figure(figsize=(12, 8))
    for region in df['who_region'].unique():
        region_data = df[df['who_region'] == region]
        yearly_region = region_data.groupby('year')['death_rate_per_100k'].mean()
        plt.plot(yearly_region.index, yearly_region.values, marker='o', linewidth=2, 
                label=region, markersize=4)
    
    plt.title('Sickle Cell Death Rate Trends by WHO Region (1990-2022)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Year', fontweight='bold', fontsize=12)
    plt.ylabel('Death Rate (per 100,000)', fontweight='bold', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regional_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return regional_stats, income_stats

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, 
                annot_kws={"size": 10, "weight": "bold"})
    plt.title('Correlation Matrix: Sickle Cell Disease Factors', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("correlation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    death_rate_correlations = corr_matrix['death_rate_per_100k'].sort_values(ascending=False)
    
    return death_rate_correlations, corr_matrix

def create_temporal_progress_chart(df):
    """Create temporal progress visualization"""
    country_progress = []
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        if len(country_data) > 1:
            early_rate = country_data[country_data['year'] == country_data['year'].min()]['death_rate_per_100k'].values[0]
            late_rate = country_data[country_data['year'] == country_data['year'].max()]['death_rate_per_100k'].values[0]
            improvement = ((early_rate - late_rate) / early_rate) * 100
            
            country_progress.append({
                'country': country,
                'early_rate': early_rate,
                'late_rate': late_rate,
                'improvement_pct': improvement,
                'who_region': country_data['who_region'].iloc[0],
                'income_level': country_data['income_level'].iloc[0]
            })
    
    progress_df = pd.DataFrame(country_progress)
    
    # Improvement by region
    plt.figure(figsize=(12, 8))
    regional_improvement = progress_df.groupby('who_region')['improvement_pct'].mean().sort_values(ascending=False)
    
    colors = ['#2A9D8F' if x > 0 else '#E63946' for x in regional_improvement]
    bars = plt.bar(regional_improvement.index, regional_improvement.values, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.title('Average Improvement in Sickle Cell Death Rates by Region (1990-2022)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Improvement (%)', fontweight='bold', fontsize=12)
    plt.xlabel('WHO Region', fontweight='bold', fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("temporal_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return progress_df

def create_healthcare_analysis_chart(df):
    """Create healthcare spending analysis chart"""
    plt.figure(figsize=(12, 8))
    
    # Bubble chart: Health spending vs death rate, sized by GDP
    scatter = plt.scatter(df['health_expenditure_pct_gdp'], df['death_rate_per_100k'],
                         s=df['gdp_per_capita_usd']/100,  # Size by GDP
                         c=df['life_expectancy'], cmap='viridis', alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label='Life Expectancy')
    plt.title('Healthcare Spending vs Mortality Rate\n(Size = GDP per Capita, Color = Life Expectancy)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Health Expenditure (% of GDP)', fontweight='bold', fontsize=12)
    plt.ylabel('Death Rate (per 100,000)', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['health_expenditure_pct_gdp'], df['death_rate_per_100k'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['health_expenditure_pct_gdp'].min(), 
                         df['health_expenditure_pct_gdp'].max(), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("healthcare_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------
# Missing Function: Generate Comprehensive Statistics
# -------------------------------
def generate_comprehensive_statistics(df):
    """Generate comprehensive statistics for the dataset"""
    stats = {
        'total_countries': df['country'].nunique(),
        'total_years': df['year'].nunique(),
        'time_period': f"{df['year'].min()}-{df['year'].max()}",
        'total_records': len(df),
        'avg_death_rate': df['death_rate_per_100k'].mean(),
        'max_death_rate': df['death_rate_per_100k'].max(),
        'min_death_rate': df['death_rate_per_100k'].min(),
        'highest_burden_country': df.loc[df['death_rate_per_100k'].idxmax(), 'country'],
        'lowest_burden_country': df.loc[df['death_rate_per_100k'].idxmin(), 'country'],
        'total_estimated_deaths': df['deaths'].sum(),
        'avg_life_expectancy': df['life_expectancy'].mean(),
        'avg_health_expenditure': df['health_expenditure_pct_gdp'].mean(),
        'avg_gdp_per_capita': df['gdp_per_capita_usd'].mean()
    }
    
    # Calculate correlations
    stats['correlation_health_spending_deaths'] = df['health_expenditure_pct_gdp'].corr(df['death_rate_per_100k'])
    stats['correlation_gdp_deaths'] = df['gdp_per_capita_usd'].corr(df['death_rate_per_100k'])
    stats['correlation_life_expectancy_deaths'] = df['life_expectancy'].corr(df['death_rate_per_100k'])
    
    return stats

# -------------------------------
# Enhanced PDF Report Generation with Images
# -------------------------------
def generate_sickle_cell_pdf(stats, country_stats, regional_stats, income_stats, 
                           correlations, progress_df, filename="Comprehensive_Sickle_Cell_Analysis_Report.pdf"):
    """Generate comprehensive sickle cell analysis PDF report with visualizations"""
    
    doc = SimpleDocTemplate(filename, pagesize=letter, 
                          topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=HexColor('#E63946'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#457B9D'),
        spaceAfter=6
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#333333'),
        spaceAfter=6
    )
    
    center_style = ParagraphStyle(
        'CenterStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#457B9D'),
        alignment=1,
        spaceAfter=6
    )
    
    # Content collection
    content = []
    
    # Title Page
    content.append(Spacer(1, 1*inch))
    content.append(Paragraph("COMPREHENSIVE SICKLE CELL DISEASE", title_style))
    content.append(Paragraph("GLOBAL ANALYSIS REPORT", title_style))
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph("Global Burden, Trends, and Strategic Recommendations", styles['Heading2']))
    content.append(Paragraph("1990-2022", styles['Heading2']))
    content.append(Spacer(1, 0.5*inch))
    
    # Add first visualization
    try:
        content.append(Image("global_trends.png", width=6*inch, height=4*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", center_style))
    content.append(Paragraph("Analysis by: Mwenda E. Njagi - GitHub.com/MwendaKE/InsightHub", center_style))
    content.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    content.append(Paragraph("Executive Summary", heading_style))
    exec_summary = f"""
    This comprehensive analysis examines global sickle cell disease patterns across {stats['total_countries']} high-burden countries 
    over a 32-year period ({stats['time_period']}). The report reveals dramatic disparities in disease burden, with death rates 
    ranging from {stats['min_death_rate']:.1f} to {stats['max_death_rate']:.1f} per 100,000 population. Strong correlations 
    between healthcare expenditure (r = {stats['correlation_health_spending_deaths']:.3f}) and economic development 
    (r = {stats['correlation_gdp_deaths']:.3f}) highlight the multifactorial nature of SCD outcomes.
    """
    content.append(Paragraph(exec_summary, normal_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Key Statistics Table
    stats_data = [
        ['Metric', 'Value', 'Significance'],
        ['Countries Analyzed', str(stats['total_countries']), 'Global coverage'],
        ['Study Period', stats['time_period'], '32-year trend analysis'],
        ['Average Death Rate', f"{stats['avg_death_rate']:.1f} per 100k", 'Global burden'],
        ['Highest Burden', f"{stats['highest_burden_country']}", f"{stats['max_death_rate']:.1f}/100k"],
        ['Lowest Burden', f"{stats['lowest_burden_country']}", f"{stats['min_death_rate']:.1f}/100k"],
        ['Total Deaths', f"{stats['total_estimated_deaths']:,.0f}", 'Cumulative impact'],
        ['Health Spending Corr', f"{stats['correlation_health_spending_deaths']:.3f}", 'Investment importance']
    ]
    
    stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#457B9D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F1FAEE')),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#A8DADC'))
    ]))
    content.append(stats_table)
    content.append(Spacer(1, 0.3*inch))
    
    # Global Trends Visualization
    content.append(Paragraph("Global Trends and Patterns", heading_style))
    content.append(Paragraph("The following charts show key global trends in sickle cell disease burden over the 32-year study period:", normal_style))
    
    try:
        content.append(Image("global_trends.png", width=6*inch, height=4.5*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Country Comparison
    content.append(Paragraph("Country-Level Analysis", heading_style))
    
    try:
        content.append(Image("top_countries.png", width=6*inch, height=4*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Economic vs Mortality Analysis
    try:
        content.append(Image("economic_vs_mortality.png", width=6*inch, height=4*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Regional Analysis
    content.append(Paragraph("Regional Disparities", heading_style))
    
    try:
        content.append(Image("regional_analysis.png", width=6*inch, height=3*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Regional Trends
    try:
        content.append(Image("regional_trends.png", width=6*inch, height=4*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Correlation Analysis
    content.append(Paragraph("Factor Correlation Analysis", heading_style))
    
    try:
        content.append(Image("correlation_analysis.png", width=5.5*inch, height=4.5*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Healthcare Analysis
    content.append(Paragraph("Healthcare Spending Impact", heading_style))
    
    try:
        content.append(Image("healthcare_analysis.png", width=6*inch, height=4*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Temporal Progress
    content.append(Paragraph("Progress Over Time", heading_style))
    
    try:
        content.append(Image("temporal_analysis.png", width=6*inch, height=4*inch))
        content.append(Spacer(1, 0.2*inch))
    except:
        pass
    
    # Sickle Cell Disease Overview
    content.append(Paragraph("Understanding Sickle Cell Disease", heading_style))
    disease_overview = """
    <b>Genetic Basis and Inheritance:</b><br/>
    Sickle Cell Disease (SCD) is an inherited blood disorder caused by a mutation in the hemoglobin-Beta gene 
    found on chromosome 11. The disease follows an autosomal recessive pattern - a child must inherit two 
    sickle cell genes (one from each parent) to have the disease. Carriers (with one gene) have sickle cell 
    trait and are generally healthy but can pass the gene to their children.<br/><br/>
    
    <b>Global Distribution:</b><br/>
    SCD predominantly affects populations from malaria-endemic regions, as the sickle cell trait provides 
    protection against malaria. This explains its high prevalence in sub-Saharan Africa, where up to 3% 
    of births may be affected by SCD. The disease also affects people of Mediterranean, Middle Eastern, 
    and South Asian ancestry.<br/><br/>
    
    <b>Clinical Impact:</b><br/>
    SCD causes chronic hemolytic anemia, pain crises, organ damage, and increased susceptibility to 
    infections. Without proper management, it can lead to significant morbidity and reduced life expectancy.
    """
    content.append(Paragraph(disease_overview, normal_style))
    
    # Strategic Recommendations
    content.append(Paragraph("Strategic Recommendations", heading_style))
    recommendations = """
    <b>1. Enhanced Prevention and Screening</b><br/>
    • Implement universal newborn screening in high-prevalence regions<br/>
    • Expand genetic counseling and carrier screening programs<br/>
    • Develop community-based awareness campaigns<br/><br/>
    
    <b>2. Healthcare System Strengthening</b><br/>
    • Train healthcare workers in comprehensive SCD management<br/>
    • Ensure consistent supply of essential medications including hydroxyurea<br/>
    • Establish specialized SCD treatment centers with multidisciplinary care<br/><br/>
    
    <b>3. Research and Innovation</b><br/>
    • Invest in affordable treatment options and point-of-care diagnostics<br/>
    • Develop gene therapy and curative approaches<br/>
    • Study genetic and environmental modifiers of disease severity<br/><br/>
    
    <b>4. Global Cooperation</b><br/>
    • Share best practices and successful intervention models<br/>
    • Coordinate international research efforts and clinical trials<br/>
    • Advocate for increased funding and political commitment
    """
    content.append(Paragraph(recommendations, normal_style))
    
    # Conclusion
    content.append(Paragraph("Conclusion", heading_style))
    conclusion = """
    This comprehensive analysis demonstrates that sickle cell disease remains a significant global health 
    challenge with profound disparities between regions and economic levels. The strong correlations 
    between healthcare investment, economic development, and disease outcomes provide clear direction 
    for future interventions. While challenges remain, the data shows that progress is achievable through 
    coordinated efforts combining healthcare strengthening, economic development, and targeted public 
    health interventions.<br/><br/>
    
    The visualizations in this report highlight both the scale of the challenge and the opportunities 
    for meaningful improvement. By implementing evidence-based strategies and learning from successful 
    interventions, substantial reductions in sickle cell disease burden are possible in the coming decades.
    """
    content.append(Paragraph(conclusion, normal_style))
    content.append(Spacer(1, 0.3*inch))
    
    # Contact Information - Centered with theme color
    contact_text = """
    This document is prepared by Mwenda E. Njagi, Program: InsightHub Analysis Program at GitHub (INSAPROG), 
    GitHub: https://github.com/MwendaKE/InsightHub Email Address: erickmwenda256@gmail.com, 
    Phone Number: +254 0702 623 729, Website: MwendaSoft.com.
    """
    
    contact_style = ParagraphStyle(
        'ContactStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#457B9D'),
        alignment=1,  # Center aligned
        spaceBefore=12,
        spaceAfter=12
    )
    
    content.append(Paragraph(contact_text, contact_style))
    
    # Build PDF
    doc.build(content)
    print(f"✅ Comprehensive Sickle Cell PDF report generated: {filename}")

# -------------------------------
# Main Analysis Function
# -------------------------------
def main():
    print("🚀 Starting Comprehensive Sickle Cell Disease Analysis...")
    
    # Load and clean data
    print("📊 Loading sickle cell data...")
    df = load_sickle_cell_data('sickle_cell_disease_global_analysis_1990_2022.csv')
    if df.empty:
        print("❌ Failed to load sickle cell data")
        return
    
    print("🧹 Cleaning and preprocessing data...")
    df_clean = clean_sickle_cell_data(df)
    if df_clean.empty:
        print("❌ No data after cleaning")
        return
    
    # Create comprehensive visualizations
    print("📊 Creating global trends charts...")
    yearly_trends = create_global_trends_chart(df_clean)
    
    print("🇺🇳 Creating country comparison charts...")
    country_stats, top_countries, bottom_countries = create_country_comparison_charts(df_clean)
    
    print("🗺️ Creating regional analysis charts...")
    regional_stats, income_stats = create_regional_analysis_charts(df_clean)
    
    print("📈 Creating correlation analysis...")
    correlations, corr_matrix = create_correlation_heatmap(df_clean)
    
    print("⏰ Creating temporal progress charts...")
    progress_df = create_temporal_progress_chart(df_clean)
    
    print("🏥 Creating healthcare analysis charts...")
    create_healthcare_analysis_chart(df_clean)
    
    print("📋 Generating comprehensive statistics...")
    stats = generate_comprehensive_statistics(df_clean)
    
    # Generate PDF report with visualizations
    print("📄 Generating comprehensive PDF report with visualizations...")
    generate_sickle_cell_pdf(stats, country_stats, regional_stats, income_stats, 
                           correlations, progress_df)
    
    # Print key insights
    print("\n" + "="*80)
    print("COMPREHENSIVE SICKLE CELL ANALYSIS - KEY INSIGHTS".center(80))
    print("="*80)
    print(f"🌍 Scope: {stats['total_countries']} countries, {stats['time_period']}")
    print(f"💀 Mortality Range: {stats['min_death_rate']:.1f} to {stats['max_death_rate']:.1f}/100k")
    print(f"📊 Highest Burden: {stats['highest_burden_country']}")
    print(f"🏥 Health Spending Correlation: r = {stats['correlation_health_spending_deaths']:.3f}")
    print(f"💰 GDP Correlation: r = {stats['correlation_gdp_deaths']:.3f}")
    print(f"📈 Visualizations: 7 comprehensive charts generated")
    print("="*80)
    
    # Cleanup temporary files
    for file in ["global_trends.png", "top_countries.png", "economic_vs_mortality.png", 
                 "regional_analysis.png", "regional_trends.png", "correlation_analysis.png", 
                 "healthcare_analysis.png", "temporal_analysis.png"]:
        try:
            os.remove(file)
        except:
            pass
    
    print("✅ Comprehensive sickle cell analysis complete! PDF report with visualizations generated successfully.")

if __name__ == "__main__":
    main()