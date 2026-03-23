import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Make sure output folder exists ───
os.makedirs('visuals', exist_ok=True)

# ─── Load real data ───
try:
    df = pd.read_csv('data/StudentsPerformance.csv')
except FileNotFoundError:
    print("Error: 'data/StudentsPerformance.csv' not found!")
    print("Please make sure the file is in the 'data' folder.")
    exit()

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.lower()

# Create useful columns
df['total_score']   = df['math_score'] + df['reading_score'] + df['writing_score']
df['average_score'] = df['total_score'] / 3
df['percentage']    = (df['total_score'] / 300 * 100).round(2)

# ─── Basic summary ───
print("=" * 70)
print(" STUDENT PERFORMANCE ANALYSIS – REAL KAGGLE DATASET")
print("=" * 70)
print(f"Number of students    : {len(df):,}")
print(f"Average Math          : {df['math_score'].mean():.1f}")
print(f"Average Reading       : {df['reading_score'].mean():.1f}")
print(f"Average Writing       : {df['writing_score'].mean():.1f}")
print(f"Overall Average Score : {df['average_score'].mean():.1f}")
print(f"Overall Average %     : {df['percentage'].mean():.1f}%")

print("\nAverage Scores by Gender:")
print(df.groupby('gender')[['math_score','reading_score','writing_score','average_score']].mean().round(1))

print("\nEffect of Test Preparation Course:")
print(df.groupby('test_preparation_course')[['math_score','reading_score','writing_score','average_score']].mean().round(1))

print("\nCount of students by gender:")
print(df['gender'].value_counts())

# ─── Visualizations ───
# Use modern seaborn style compatible with recent matplotlib
sns.set_theme(style="whitegrid")          # recommended & clean look
# alternative: plt.style.use('seaborn-v0_8-whitegrid')

# ── 1. Barplot – Average scores by subject & gender ──
plt.figure(figsize=(10, 6))
sns.barplot(
    x='gender',
    y='value',
    hue='variable',
    data=pd.melt(
        df,
        id_vars=['gender'],
        value_vars=['math_score', 'reading_score', 'writing_score']
    ),
    palette='Set2'
)
plt.title('Average Scores by Subject and Gender', fontsize=14, pad=12)
plt.ylabel('Score (0–100)')
plt.xlabel('Gender')
plt.legend(title='Subject')
plt.tight_layout()
plt.savefig('visuals/1_gender_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# ── 2. Boxplot – Average score by lunch type ──
plt.figure(figsize=(9, 6))
sns.boxplot(
    x='lunch',
    y='average_score',
    data=df,
    palette='Pastel1'
)
plt.title('Average Score Distribution by Lunch Type', fontsize=14, pad=12)
plt.ylabel('Average Score')
plt.xlabel('Lunch Type')
plt.tight_layout()
plt.savefig('visuals/2_lunch_impact.png', dpi=200, bbox_inches='tight')
plt.close()

# ── 3. Heatmap – Correlation between scores ──
plt.figure(figsize=(8, 6))
corr = df[['math_score', 'reading_score', 'writing_score', 'average_score']].corr()
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    cbar_kws={'label': 'Correlation'}
)
plt.title('Correlation Between Different Scores', fontsize=14, pad=12)
plt.tight_layout()
plt.savefig('visuals/3_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

# ── 4. Bonus: Count plot – Gender vs Test Preparation ──
plt.figure(figsize=(9, 5.5))
sns.countplot(
    x='gender',
    hue='test_preparation_course',
    data=df,
    palette='Set3'
)
plt.title('Students by Gender and Test Preparation Status', fontsize=14, pad=12)
plt.xlabel('Gender')
plt.ylabel('Number of Students')
plt.legend(title='Test Preparation')
plt.tight_layout()
plt.savefig('visuals/4_gender_vs_testprep.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved successfully in folder: visuals/")
print("Files created:")
print("  1_gender_comparison.png")
print("  2_lunch_impact.png")
print("  3_correlation_heatmap.png")
print("  4_gender_vs_testprep.png")
print("\nYou can now open these .png files to include in your portfolio / report.")