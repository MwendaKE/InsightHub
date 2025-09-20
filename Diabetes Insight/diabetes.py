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
def load_diabetes_data(file_path):
    """Load diabetes data from local CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Diabetes data loaded: {len(df)} records, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"❌ Error loading diabetes data: {e}")
        return pd.DataFrame()

def clean_diabetes_data(df):
    """Clean and preprocess diabetes data"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Convert all numeric columns
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop rows with missing critical data
    df_clean = df_clean.dropna(subset=['Glucose', 'BMI', 'Age', 'Outcome'])
    
    print(f"✅ Data cleaned: {len(df_clean)} records remaining")
    return df_clean

# -------------------------------
# Enhanced Analysis Functions with Better Image Formatting
# -------------------------------
def feature_correlation_analysis(df):
    """Comprehensive feature correlation analysis with better image formatting"""
    corr_matrix = df.corr()
    outcome_corr = corr_matrix['Outcome'].sort_values(ascending=False)
    
    # Top and bottom features
    top_5 = outcome_corr.head(6)[1:6]  # Skip Outcome itself
    bottom_5 = outcome_corr.tail(5)
    
    # Plot top features - tightly cropped
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, len(top_5)))
    bars = plt.barh(top_5.index, top_5.values, color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.title('Top 5 Features Correlated with Diabetes Outcome', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Correlation Coefficient', fontsize=10)
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=1.0)
    plt.savefig("top_features.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return corr_matrix, outcome_corr

def outcome_distribution_analysis(df):
    """Analyze outcome distribution"""
    outcome_stats = {
        'diabetes_count': df['Outcome'].sum(),
        'non_diabetes_count': len(df) - df['Outcome'].sum(),
        'diabetes_percentage': (df['Outcome'].sum() / len(df)) * 100,
        'avg_glucose_diabetes': df[df['Outcome'] == 1]['Glucose'].mean(),
        'avg_glucose_non_diabetes': df[df['Outcome'] == 0]['Glucose'].mean(),
        'avg_bmi_diabetes': df[df['Outcome'] == 1]['BMI'].mean(),
        'avg_bmi_non_diabetes': df[df['Outcome'] == 0]['BMI'].mean()
    }
    
    # Plot distribution - tightly cropped
    plt.figure(figsize=(10, 6))
    outcome_counts = df['Outcome'].value_counts()
    colors = ['#4ECDC4', '#FF6B6B']
    plt.bar(['Non-Diabetic', 'Diabetic'], outcome_counts.values, color=colors, alpha=0.8)
    
    for i, count in enumerate(outcome_counts.values):
        plt.text(i, count + 5, f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Diabetes Outcome Distribution', fontsize=14, fontweight='bold', pad=10)
    plt.ylabel('Number of Patients', fontsize=10)
    plt.tight_layout(pad=1.0)
    plt.savefig("outcome_distribution.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return outcome_stats

def feature_relationships_analysis(df, top_features):
    """Analyze relationships between top features and outcome"""
    # Create scatter plots for top 4 features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    colors = ['#4ECDC4', '#FF6B6B']  # Non-diabetic, Diabetic
    
    for i, feature in enumerate(top_features[:4]):
        for outcome in [0, 1]:
            subset = df[df['Outcome'] == outcome]
            axes[i].scatter(subset[feature], subset['Glucose'], alpha=0.6, s=20, color=colors[outcome], 
                           label='Diabetic' if outcome == 1 else 'Non-Diabetic')
        
        corr = df[feature].corr(df['Outcome'])
        axes[i].set_title(f'{feature} vs Glucose (r={corr:.3f})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel('Glucose Level', fontsize=10)
        
        if i == 0:
            axes[i].legend()
    
    plt.tight_layout(pad=2.0)
    plt.savefig("feature_relationships.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def glucose_analysis(df):
    """Analyze glucose levels by outcome"""
    plt.figure(figsize=(10, 6))
    
    # Box plot of glucose by outcome
    diabetic_glucose = df[df['Outcome'] == 1]['Glucose']
    non_diabetic_glucose = df[df['Outcome'] == 0]['Glucose']
    
    box_data = [non_diabetic_glucose, diabetic_glucose]
    box = plt.boxplot(box_data, labels=['Non-Diabetic', 'Diabetic'], patch_artist=True)
    
    colors = ['#4ECDC4', '#FF6B6B']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Glucose Levels by Diabetes Outcome', fontsize=14, fontweight='bold', pad=10)
    plt.ylabel('Glucose Level', fontsize=10)
    plt.tight_layout(pad=1.0)
    plt.savefig("glucose_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return {
        'glucose_diabetic_mean': diabetic_glucose.mean(),
        'glucose_non_diabetic_mean': non_diabetic_glucose.mean(),
        'glucose_difference': diabetic_glucose.mean() - non_diabetic_glucose.mean()
    }

def bmi_analysis(df):
    """Analyze BMI by outcome"""
    plt.figure(figsize=(10, 6))
    
    diabetic_bmi = df[df['Outcome'] == 1]['BMI']
    non_diabetic_bmi = df[df['Outcome'] == 0]['BMI']
    
    box_data = [non_diabetic_bmi, diabetic_bmi]
    box = plt.boxplot(box_data, labels=['Non-Diabetic', 'Diabetic'], patch_artist=True)
    
    colors = ['#4ECDC4', '#FF6B6B']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('BMI by Diabetes Outcome', fontsize=14, fontweight='bold', pad=10)
    plt.ylabel('BMI', fontsize=10)
    plt.tight_layout(pad=1.0)
    plt.savefig("bmi_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return {
        'bmi_diabetic_mean': diabetic_bmi.mean(),
        'bmi_non_diabetic_mean': non_diabetic_bmi.mean(),
        'bmi_difference': diabetic_bmi.mean() - non_diabetic_bmi.mean()
    }

def generate_diabetes_statistics(df, outcome_stats, glucose_stats, bmi_stats, top_features):
    """Generate comprehensive statistics"""
    stats = {
        'total_patients': len(df),
        'total_features': len(df.columns) - 1,  # exclude Outcome
        'diabetes_percentage': outcome_stats['diabetes_percentage'],
        'diabetes_count': outcome_stats['diabetes_count'],
        'non_diabetes_count': outcome_stats['non_diabetes_count'],
        'glucose_difference': glucose_stats['glucose_difference'],
        'bmi_difference': bmi_stats['bmi_difference'],
        'top_features': top_features,
        'feature_correlations': df.corr()['Outcome'].sort_values(ascending=False).to_dict(),
        'avg_age': df['Age'].mean(),
        'avg_pregnancies': df['Pregnancies'].mean()
    }
    
    return stats

# -------------------------------
# Enhanced PDF Report Generation
# -------------------------------
def generate_diabetes_pdf(corr_matrix, outcome_stats, glucose_stats, bmi_stats, stats, filename="Diabetes_Analysis_Report.pdf"):
    """Generate comprehensive diabetes analysis PDF report"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Colors
    primary_color = HexColor('#2E86AB')  # Diabetes blue
    secondary_color = HexColor('#A23B72')  # Purple
    accent_color = HexColor('#F18F01')    # Orange
    
    def add_footer():
        """Add footer to current page"""
        c.setFillColor(HexColor('#666666'))
        c.setFont("Helvetica-Oblique", 8)
        footer_text = "Generated by Insight Hub Analysis Program created by Mwenda E. Njagi at GitHub.com. Link: https://github.com/MwendaKE/InsightHub."
        c.drawCentredString(width/2, 20, footer_text)
    
    def draw_text_lines(lines, start_y, line_height=15, left_margin=70, right_margin=50, font_name="Helvetica", font_size=10, text_color=HexColor('#333333')):
        """Helper function to draw text lines with automatic pagination"""
        current_y = start_y
        c.setFont(font_name, font_size)
        c.setFillColor(text_color)
        
        for line in lines:
            if current_y < 50:
                add_footer()
                c.showPage()
                current_y = height - 50
                c.setFont(font_name, font_size)
                c.setFillColor(text_color)
            c.drawString(left_margin, current_y, line)
            current_y -= line_height
        return current_y
    
    # Title Page
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-100, "DIABETES PATIENT ANALYSIS REPORT")
    
    c.setFillColor(secondary_color)
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-150, "Comprehensive Clinical Feature Analysis")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-200, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    c.setFillColor(accent_color)
    c.setFont("Helvetica-Oblique", 13)
    c.drawCentredString(width/2, height-250, "Analysed by Mwenda E. Njagi @ Github.com/MwendaKE/InsightHub")
    
    c.setFillColor(HexColor('#666666'))
    c.setFont("Helvetica", 11)
    c.drawCentredString(width/2, height-300, f"Data Source: Local Diabetes Dataset ({stats['total_patients']} patients, {stats['total_features']} features)")
    
    add_footer()
    c.showPage()
    
    # Executive Summary
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-50, "Executive Summary")
    
    summary_text = [
        f"• Comprehensive analysis of {stats['total_patients']} patient records",
        f"• {stats['total_features']} clinical features analyzed",
        f"• Diabetes prevalence: {stats['diabetes_percentage']:.1f}% ({stats['diabetes_count']} patients)",
        f"• Non-diabetic: {stats['non_diabetes_count']} patients",
        f"• Glucose difference: +{stats['glucose_difference']:.1f} mg/dL in diabetic patients",
        f"• BMI difference: +{stats['bmi_difference']:.1f} in diabetic patients",
        f"• Average age: {stats['avg_age']:.1f} years",
        f"• Average pregnancies: {stats['avg_pregnancies']:.1f}",
        "",
        "Key Insights:",
        "• Strong correlations between clinical features and diabetes outcome",
        "• Significant glucose and BMI differences between groups",
        "• Multiple features show predictive power for diabetes risk",
        "• Potential for early detection using clinical markers"
    ]
    
    y_pos = height - 80
    y_pos = draw_text_lines(summary_text, y_pos)
    
    add_footer()
    c.showPage()
    
    # Outcome Distribution
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Diabetes Outcome Distribution")
    c.drawImage("outcome_distribution.png", 50, height-330, width=500, height=250)
    
    # Outcome insights
    outcome_text = [
        "Patient Distribution:",
        f"• Diabetic patients: {stats['diabetes_count']} ({stats['diabetes_percentage']:.1f}%)",
        f"• Non-diabetic patients: {stats['non_diabetes_count']}",
        f"• Overall prevalence: {stats['diabetes_percentage']:.1f}%",
        "",
        "Clinical Significance:",
        "• Balanced dataset for analysis",
        "• Sufficient cases for meaningful insights",
        "• Representative sample for diabetes research"
    ]
    
    y_outcome = height - 600
    y_outcome = draw_text_lines(outcome_text, y_outcome)
    
    add_footer()
    c.showPage()
    
    # Feature Correlation Analysis
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Feature Correlation Analysis")
    c.drawImage("top_features.png", 50, height-280, width=500, height=200)
    
    # Correlation insights
    corr_text = [
        "Top Predictive Features:",
        f"• {stats['top_features'][0]}: {stats['feature_correlations'][stats['top_features'][0]]:.3f}",
        f"• {stats['top_features'][1]}: {stats['feature_correlations'][stats['top_features'][1]]:.3f}",
        f"• {stats['top_features'][2]}: {stats['feature_correlations'][stats['top_features'][2]]:.3f}",
        f"• {stats['top_features'][3]}: {stats['feature_correlations'][stats['top_features'][3]]:.3f}",
        f"• {stats['top_features'][4]}: {stats['feature_correlations'][stats['top_features'][4]]:.3f}",
        "",
        "Interpretation:",
        "• Values closer to ±1 indicate stronger relationships",
        "• Positive values: feature increase = diabetes risk increase",
        "• Negative values: feature increase = diabetes risk decrease"
    ]
    
    y_corr = height - 500
    y_corr = draw_text_lines(corr_text, y_corr)
    
    add_footer()
    c.showPage()
    
    # Glucose Analysis
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Glucose Level Analysis")
    c.drawImage("glucose_analysis.png", 50, height-330, width=500, height=250)
    
    # Glucose insights
    glucose_text = [
        "Glucose Statistics:",
        f"• Diabetic average: {glucose_stats['glucose_diabetic_mean']:.1f} mg/dL",
        f"• Non-diabetic average: {glucose_stats['glucose_non_diabetic_mean']:.1f} mg/dL",
        f"• Difference: +{glucose_stats['glucose_difference']:.1f} mg/dL",
        "",
        "Clinical Significance:",
        "• Clear separation between groups",
        "• Glucose is strong diabetes predictor",
        "• Monitoring glucose crucial for diagnosis"
    ]
    
    y_glucose = height - 600
    y_glucose = draw_text_lines(glucose_text, y_glucose)
    
    add_footer()
    c.showPage()
    
    # BMI Analysis
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "BMI Analysis")
    c.drawImage("bmi_analysis.png", 50, height-330, width=500, height=250)
    
    # BMI insights
    bmi_text = [
        "BMI Statistics:",
        f"• Diabetic average: {bmi_stats['bmi_diabetic_mean']:.1f}",
        f"• Non-diabetic average: {bmi_stats['bmi_non_diabetic_mean']:.1f}",
        f"• Difference: +{bmi_stats['bmi_difference']:.1f}",
        "",
        "Clinical Significance:",
        "• BMI strongly associated with diabetes risk",
        "• Weight management important for prevention",
        "• Lifestyle factors play significant role"
    ]
    
    y_bmi = height - 600
    y_bmi = draw_text_lines(bmi_text, y_bmi)
    
    add_footer()
    c.showPage()
    
    # Feature Relationships
    c.setFillColor(secondary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Feature Relationships")
    c.drawImage("feature_relationships.png", 50, height-380, width=500, height=300)
    
    # Relationship insights
    rel_text = [
        "Relationship Analysis:",
        "• Complex interactions between features",
        "• Some features show clear separation",
        "• Others demonstrate overlapping patterns",
        "",
        "Clinical Implications:",
        "• Multiple factors contribute to diabetes risk",
        "• Comprehensive assessment needed",
        "• Personalized risk evaluation important"
    ]
    
    y_rel = height - 700
    y_rel = draw_text_lines(rel_text, y_rel)
    
    add_footer()
    c.showPage()
    
    # Recommendations
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Clinical Recommendations & Insights")
    
    recommendations = [
        "1. RISK ASSESSMENT:",
        "   • Focus on patients with high glucose levels (>126 mg/dL)",
        "   • Monitor individuals with BMI > 30 closely",
        "   • Consider age and pregnancy history in assessment",
        "",
        "2. PREVENTION STRATEGIES:",
        "   • Weight management programs for high-BMI individuals",
        "   • Regular glucose monitoring for at-risk patients",
        "   • Lifestyle modification education",
        "",
        "3. EARLY DETECTION:",
        "   • Regular screening for patients with multiple risk factors",
        "   • Use feature correlations for risk stratification",
        "   • Implement predictive modeling for early intervention",
        "",
        "4. PATIENT EDUCATION:",
        "   • Educate about diabetes risk factors",
        "   • Promote healthy eating and exercise",
        "   • Regular health check-ups",
        "",
        "5. DATA-DRIVEN CARE:",
        "   • Continuous monitoring of clinical markers",
        "   • Personalized risk assessment",
        "   • Evidence-based treatment decisions"
    ]
    
    y_rec = height - 80
    y_rec = draw_text_lines(recommendations, y_rec, line_height=15)
    
    add_footer()
    c.save()
    print(f"✅ Diabetes PDF report generated: {filename}")

# -------------------------------
# Main Function
# -------------------------------
def main():
    print("🚀 Starting Diabetes Patient Data Analysis...")
    
    # Load and clean data
    print("📊 Loading diabetes data from local file...")
    df = load_diabetes_data("../Data Sets/diabetes.csv")
    if df.empty:
        print("❌ Failed to load diabetes data")
        return
    
    print("🧹 Cleaning and preprocessing data...")
    df_clean = clean_diabetes_data(df)
    if df_clean.empty:
        print("❌ No data after cleaning")
        return
    
    # Comprehensive analysis
    print("📈 Analyzing feature correlations...")
    corr_matrix, outcome_corr = feature_correlation_analysis(df_clean)
    top_features = outcome_corr.index[1:6].tolist()  # Top 5 features excluding Outcome
    
    print("📊 Analyzing outcome distribution...")
    outcome_stats = outcome_distribution_analysis(df_clean)
    
    print("🩸 Analyzing glucose levels...")
    glucose_stats = glucose_analysis(df_clean)
    
    print("⚖️ Analyzing BMI patterns...")
    bmi_stats = bmi_analysis(df_clean)
    
    print("🔗 Analyzing feature relationships...")
    feature_relationships_analysis(df_clean, top_features)
    
    print("📋 Generating comprehensive statistics...")
    stats = generate_diabetes_statistics(df_clean, outcome_stats, glucose_stats, bmi_stats, top_features)
    
    # Generate PDF report
    print("📄 Generating comprehensive PDF report...")
    generate_diabetes_pdf(corr_matrix, outcome_stats, glucose_stats, bmi_stats, stats)
    
    # Print key insights
    print("\n" + "="*70)
    print("DIABETES ANALYSIS - KEY INSIGHTS".center(70))
    print("="*70)
    print(f"📊 Patients: {stats['total_patients']}, Features: {stats['total_features']}")
    print(f"🎯 Diabetes prevalence: {stats['diabetes_percentage']:.1f}%")
    print(f"🩸 Glucose difference: +{stats['glucose_difference']:.1f} mg/dL")
    print(f"⚖️ BMI difference: +{stats['bmi_difference']:.1f}")
    print(f"🔝 Top Feature: {top_features[0]} (r={outcome_corr[top_features[0]]:.3f})")
    print(f"📋 Top 5 Features: {', '.join(top_features)}")
    print("="*70)
    
    # Cleanup
    for file in ["top_features.png", "outcome_distribution.png", "feature_relationships.png", 
                 "glucose_analysis.png", "bmi_analysis.png"]:
        try:
            os.remove(file)
        except:
            pass
    
    print("✅ Diabetes analysis complete! Report generated successfully.")

if __name__ == "__main__":
    main()