import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("pastel")

# -------------------------------
# Data Loading and Cleaning Functions
# -------------------------------
def load_titanic_data(file_path):
    """Load Titanic data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Titanic data loaded: {len(df)} passengers, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading Titanic data: {e}")
        return pd.DataFrame()

def clean_titanic_data(df):
    """Clean and preprocess Titanic data"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Fill missing Age values with median age
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    
    # Fill missing Embarked with mode
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    
    # Fill missing Fare with median fare
    df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
    
    # Create new features
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['IsAlone'] = df_clean['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    
    # Extract titles from names
    df_clean['Title'] = df_clean['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_clean['Title'] = df_clean['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_clean['Title'] = df_clean['Title'].replace('Mlle', 'Miss')
    df_clean['Title'] = df_clean['Title'].replace('Ms', 'Miss')
    df_clean['Title'] = df_clean['Title'].replace('Mme', 'Mrs')
    
    # Create age groups
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                 labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    print(f"✅ Data cleaned: {len(df_clean)} passengers")
    return df_clean

# -------------------------------
# Analysis Functions
# -------------------------------
def survival_by_class(df):
    """Analyze survival by passenger class"""
    class_survival = df.groupby('Pclass')['Survived'].agg(['mean', 'count']).reset_index()
    class_survival.columns = ['Pclass', 'SurvivalRate', 'Count']
    class_survival['SurvivalRate'] = class_survival['SurvivalRate'] * 100
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Pclass', y='SurvivalRate', data=class_survival, alpha=0.8)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{class_survival["SurvivalRate"][i]:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                   fontweight='bold')
    
    plt.title('Survival Rate by Passenger Class', fontsize=16, fontweight='bold')
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("survival_by_class.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return class_survival

def survival_by_gender(df):
    """Analyze survival by gender"""
    gender_survival = df.groupby('Sex')['Survived'].agg(['mean', 'count']).reset_index()
    gender_survival.columns = ['Sex', 'SurvivalRate', 'Count']
    gender_survival['SurvivalRate'] = gender_survival['SurvivalRate'] * 100
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Sex', y='SurvivalRate', data=gender_survival, alpha=0.8)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{gender_survival["SurvivalRate"][i]:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                   fontweight='bold')
    
    plt.title('Survival Rate by Gender', fontsize=16, fontweight='bold')
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("survival_by_gender.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return gender_survival

def survival_by_age(df):
    """Analyze survival by age groups"""
    age_survival = df.groupby('AgeGroup')['Survived'].agg(['mean', 'count']).reset_index()
    age_survival.columns = ['AgeGroup', 'SurvivalRate', 'Count']
    age_survival['SurvivalRate'] = age_survival['SurvivalRate'] * 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='AgeGroup', y='SurvivalRate', data=age_survival, alpha=0.8)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{age_survival["SurvivalRate"][i]:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                   fontweight='bold')
    
    plt.title('Survival Rate by Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("survival_by_age.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return age_survival

def survival_by_family(df):
    """Analyze survival by family size"""
    family_survival = df.groupby('IsAlone')['Survived'].agg(['mean', 'count']).reset_index()
    family_survival.columns = ['IsAlone', 'SurvivalRate', 'Count']
    family_survival['IsAlone'] = family_survival['IsAlone'].map({0: 'With Family', 1: 'Alone'})
    family_survival['SurvivalRate'] = family_survival['SurvivalRate'] * 100
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='IsAlone', y='SurvivalRate', data=family_survival, alpha=0.8)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{family_survival["SurvivalRate"][i]:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                   fontweight='bold')
    
    plt.title('Survival Rate by Family Status', fontsize=16, fontweight='bold')
    plt.xlabel('Family Status', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("survival_by_family.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return family_survival

def fare_distribution(df):
    """Analyze fare distribution by class and survival"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df)
    plt.title('Fare Distribution by Class and Survival', fontsize=16, fontweight='bold')
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Fare', fontsize=12)
    plt.legend(title='Survived', loc='upper right')
    plt.tight_layout()
    plt.savefig("fare_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def embarked_analysis(df):
    """Analyze survival by embarkation port"""
    embarked_survival = df.groupby('Embarked')['Survived'].agg(['mean', 'count']).reset_index()
    embarked_survival.columns = ['Embarked', 'SurvivalRate', 'Count']
    embarked_survival['SurvivalRate'] = embarked_survival['SurvivalRate'] * 100
    
    # Map embarkation codes to full names
    embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    embarked_survival['Embarked'] = embarked_survival['Embarked'].map(embarked_map)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Embarked', y='SurvivalRate', data=embarked_survival, alpha=0.8)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{embarked_survival["SurvivalRate"][i]:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                   fontweight='bold')
    
    plt.title('Survival Rate by Embarkation Port', fontsize=16, fontweight='bold')
    plt.xlabel('Embarkation Port', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("survival_by_embarked.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return embarked_survival

def generate_statistics(df):
    """Generate comprehensive statistics"""
    stats = {
        'total_passengers': len(df),
        'survival_rate': df['Survived'].mean() * 100,
        'male_passengers': len(df[df['Sex'] == 'male']),
        'female_passengers': len(df[df['Sex'] == 'female']),
        'male_survival_rate': df[df['Sex'] == 'male']['Survived'].mean() * 100,
        'female_survival_rate': df[df['Sex'] == 'female']['Survived'].mean() * 100,
        'first_class_passengers': len(df[df['Pclass'] == 1]),
        'second_class_passengers': len(df[df['Pclass'] == 2]),
        'third_class_passengers': len(df[df['Pclass'] == 3]),
        'first_class_survival_rate': df[df['Pclass'] == 1]['Survived'].mean() * 100,
        'second_class_survival_rate': df[df['Pclass'] == 2]['Survived'].mean() * 100,
        'third_class_survival_rate': df[df['Pclass'] == 3]['Survived'].mean() * 100,
        'average_age': df['Age'].mean(),
        'average_fare': df['Fare'].mean(),
        'children_count': len(df[df['Age'] < 18]),
        'children_survival_rate': df[df['Age'] < 18]['Survived'].mean() * 100,
        'alone_passengers': len(df[df['IsAlone'] == 1]),
        'alone_survival_rate': df[df['IsAlone'] == 1]['Survived'].mean() * 100,
        'family_survival_rate': df[df['IsAlone'] == 0]['Survived'].mean() * 100
    }
    
    return stats

# -------------------------------
# PDF Report Generation
# -------------------------------
def generate_titanic_pdf(class_data, gender_data, age_data, family_data, embarked_data, stats, filename="Titanic_Analysis_Report.pdf"):
    """Generate Titanic analysis PDF report"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Colors
    primary_color = HexColor('#1E4D79')  # Deep blue
    secondary_color = HexColor('#5B8CA8')  # Light blue
    accent_color = HexColor('#A8DADC')  # Very light blue
    
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
    c.drawCentredString(width/2, height-100, "TITANIC SURVIVAL ANALYSIS REPORT")
    
    c.setFillColor(secondary_color)
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-150, "What Factors Influenced Survival?")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-200, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-250, f"Analysed by Mwenda E. Njagi @ Github.com/MwendaKE/InsightHub")
    
    c.setFillColor(HexColor('#666666'))
    c.setFont("Helvetica", 11)
    c.drawCentredString(width/2, height-300, "Data Source: Titanic Passenger Dataset")
    
    add_footer()
    c.showPage()
    
    # Executive Summary
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-50, "Executive Summary")
    
    summary_text = [
        f"• Analysis of {stats['total_passengers']} passengers aboard the Titanic",
        f"• Overall survival rate: {stats['survival_rate']:.1f}%",
        f"• Female survival rate: {stats['female_survival_rate']:.1f}%",
        f"• Male survival rate: {stats['male_survival_rate']:.1f}%",
        f"• 1st class survival rate: {stats['first_class_survival_rate']:.1f}%",
        f"• 3rd class survival rate: {stats['third_class_survival_rate']:.1f}%",
        f"• Children survival rate: {stats['children_survival_rate']:.1f}%",
        f"• Average age: {stats['average_age']:.1f} years",
        f"• Average fare: ${stats['average_fare']:.2f}",
        "",
        "Key Insights:",
        "• 'Women and children first' protocol was followed",
        "• Higher socioeconomic status improved survival chances",
        "• Traveling with family increased survival probability"
    ]
    
    y_pos = height - 80
    y_pos = draw_text_lines(summary_text, y_pos)
    
    add_footer()
    c.showPage()
    
    # Class Analysis
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Survival by Passenger Class")
    c.drawImage("survival_by_class.png", 50, height-280, width=500, height=200)
    
    class_text = [
        "Class Analysis:",
        f"• 1st Class: {stats['first_class_passengers']} passengers, {stats['first_class_survival_rate']:.1f}% survived",
        f"• 2nd Class: {stats['second_class_passengers']} passengers, {stats['second_class_survival_rate']:.1f}% survived",
        f"• 3rd Class: {stats['third_class_passengers']} passengers, {stats['third_class_survival_rate']:.1f}% survived",
        "",
        "Key Finding:",
        "• 1st class passengers had 2.5x higher survival rate than 3rd class"
    ]
    
    y_class = height - 500
    y_class = draw_text_lines(class_text, y_class)
    
    add_footer()
    c.showPage()
    
    # Gender Analysis
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Survival by Gender")
    c.drawImage("survival_by_gender.png", 50, height-280, width=500, height=200)
    
    gender_text = [
        "Gender Analysis:",
        f"• Female passengers: {stats['female_passengers']}, {stats['female_survival_rate']:.1f}% survived",
        f"• Male passengers: {stats['male_passengers']}, {stats['male_survival_rate']:.1f}% survived",
        "",
        "Key Finding:",
        "• Women had 3.5x higher survival rate than men"
    ]
    
    y_gender = height - 500
    y_gender = draw_text_lines(gender_text, y_gender)
    
    add_footer()
    c.showPage()
    
    # Age Analysis
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Survival by Age Group")
    c.drawImage("survival_by_age.png", 50, height-280, width=500, height=200)
    
    age_text = [
        "Age Analysis:",
        f"• Children (<18): {stats['children_count']} passengers, {stats['children_survival_rate']:.1f}% survived",
        f"• Average age: {stats['average_age']:.1f} years",
        "",
        "Key Finding:",
        "• The 'children first' protocol was followed"
    ]
    
    y_age = height - 500
    y_age = draw_text_lines(age_text, y_age)
    
    add_footer()
    c.showPage()
    
    # Family Analysis
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Survival by Family Status")
    c.drawImage("survival_by_family.png", 50, height-280, width=500, height=200)
    
    family_text = [
        "Family Analysis:",
        f"• Passengers traveling alone: {stats['alone_passengers']}, {stats['alone_survival_rate']:.1f}% survived",
        f"• Passengers with family: {stats['total_passengers'] - stats['alone_passengers']}, {stats['family_survival_rate']:.1f}% survived",
        "",
        "Key Finding:",
        "• Traveling with family increased survival chances"
    ]
    
    y_family = height - 500
    y_family = draw_text_lines(family_text, y_family)
    
    add_footer()
    c.showPage()
    
    # Fare Distribution
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Fare Distribution by Class and Survival")
    c.drawImage("fare_distribution.png", 50, height-380, width=500, height=300)
    
    fare_text = [
        "Fare Analysis:",
        f"• Average fare: ${stats['average_fare']:.2f}",
        "",
        "Key Finding:",
        "• Higher fares (correlated with higher class) were associated with better survival rates"
    ]
    
    y_fare = height - 700
    y_fare = draw_text_lines(fare_text, y_fare)
    
    add_footer()
    c.showPage()
    
    # Embarkation Analysis
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Survival by Embarkation Port")
    c.drawImage("survival_by_embarked.png", 50, height-280, width=500, height=200)
    
    embarked_text = [
        "Embarkation Analysis:",
        f"• Cherbourg: {embarked_data[embarked_data['Embarked'] == 'Cherbourg']['Count'].values[0]} passengers, {embarked_data[embarked_data['Embarked'] == 'Cherbourg']['SurvivalRate'].values[0]:.1f}% survived",
        f"• Queenstown: {embarked_data[embarked_data['Embarked'] == 'Queenstown']['Count'].values[0]} passengers, {embarked_data[embarked_data['Embarked'] == 'Queenstown']['SurvivalRate'].values[0]:.1f}% survived",
        f"• Southampton: {embarked_data[embarked_data['Embarked'] == 'Southampton']['Count'].values[0]} passengers, {embarked_data[embarked_data['Embarked'] == 'Southampton']['SurvivalRate'].values[0]:.1f}% survived",
        "",
        "Key Finding:",
        "• Passengers from Cherbourg had higher survival rates"
    ]
    
    y_embarked = height - 500
    y_embarked = draw_text_lines(embarked_text, y_embarked)
    
    add_footer()
    c.showPage()
    
    # Conclusion
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Conclusion & Key Takeaways")
    
    conclusion = [
        "The Titanic disaster revealed significant patterns in survival:",
        "",
        "1. SOCIOECONOMIC FACTORS:",
        "   • Higher class passengers had significantly better survival rates",
        "   • Wealth and status provided access to better locations on the ship",
        "",
        "2. GENDER AND AGE PRIORITIES:",
        "   • The 'women and children first' protocol was largely followed",
        "   • Female survival rate was 3.5x higher than male survival rate",
        "",
        "3. FAMILY SUPPORT:",
        "   • Traveling with family members increased survival chances",
        "   • Alone passengers had lower survival rates",
        "",
        "4. HISTORICAL CONTEXT:",
        "   • These patterns reflect early 20th century social norms and values",
        "   • The disaster led to improved maritime safety regulations"
    ]
    
    y_conc = height - 80
    y_conc = draw_text_lines(conclusion, y_conc, line_height=15)
    
    add_footer()
    c.save()
    print(f"✅ Titanic PDF report generated: {filename}")

# -------------------------------
# Main Function
# -------------------------------
def main():
    print("🚀 Starting Titanic Data Analysis...")
    
    # Load and clean data
    print("📊 Loading Titanic data...")
    df = load_titanic_data("../Data Sets/titanic.csv")
    if df.empty:
        print("❌ Failed to load Titanic data")
        return
    
    print("🧹 Cleaning and preprocessing data...")
    df_clean = clean_titanic_data(df)
    if df_clean.empty:
        print("❌ No data after cleaning")
        return
    
    # Comprehensive analysis
    print("📈 Analyzing survival by class...")
    class_data = survival_by_class(df_clean)
    
    print("📈 Analyzing survival by gender...")
    gender_data = survival_by_gender(df_clean)
    
    print("📈 Analyzing survival by age...")
    age_data = survival_by_age(df_clean)
    
    print("📈 Analyzing survival by family...")
    family_data = survival_by_family(df_clean)
    
    print("📈 Analyzing fare distribution...")
    fare_distribution(df_clean)
    
    print("📈 Analyzing survival by embarkation port...")
    embarked_data = embarked_analysis(df_clean)
    
    print("📋 Generating comprehensive statistics...")
    stats = generate_statistics(df_clean)
    
    # Generate PDF report
    print("📄 Generating PDF report...")
    generate_titanic_pdf(class_data, gender_data, age_data, family_data, embarked_data, stats)
    
    # Print key insights
    print("\n" + "="*70)
    print("TITANIC ANALYSIS - KEY INSIGHTS".center(70))
    print("="*70)
    print(f"📊 Overall Survival: {stats['survival_rate']:.1f}%")
    print(f"👩 Women Survival: {stats['female_survival_rate']:.1f}%")
    print(f"👨 Men Survival: {stats['male_survival_rate']:.1f}%")
    print(f"🥇 1st Class Survival: {stats['first_class_survival_rate']:.1f}%")
    print(f"🥉 3rd Class Survival: {stats['third_class_survival_rate']:.1f}%")
    print(f"👶 Children Survival: {stats['children_survival_rate']:.1f}%")
    print(f"👥 Alone Survival: {stats['alone_survival_rate']:.1f}%")
    print(f"👨‍👩‍👧‍👦 Family Survival: {stats['family_survival_rate']:.1f}%")
    print("="*70)
    
    # Cleanup
    for file in ["survival_by_class.png", "survival_by_gender.png", "survival_by_age.png", 
                 "survival_by_family.png", "fare_distribution.png", "survival_by_embarked.png"]:
        try:
            os.remove(file)
        except:
            pass
    
    print("✅ Titanic analysis complete! Report generated successfully.")

if __name__ == "__main__":
    main()