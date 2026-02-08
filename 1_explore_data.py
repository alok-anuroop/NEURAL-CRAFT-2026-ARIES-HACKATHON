import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from config import config

# Set style
sns.set_style('whitegrid')

# Load data
print("Loading data...")
train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Display first few rows
print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)
print(train_df.head())

# Check for missing values
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(train_df.isnull().sum())

# Text length analysis
print("\n" + "="*80)
print("TEXT LENGTH STATISTICS")
print("="*80)
train_df['text_length'] = train_df['complaint_text'].str.len()
train_df['word_count'] = train_df['complaint_text'].str.split().str.len()

print(train_df[['text_length', 'word_count']].describe())

# Visualize text length distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(train_df['text_length'], bins=50, edgecolor='black')
plt.xlabel('Character Count')
plt.ylabel('Frequency')
plt.title('Distribution of Complaint Text Length')
plt.axvline(train_df['text_length'].median(), color='red', linestyle='--', label=f'Median: {train_df["text_length"].median():.0f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(train_df['word_count'], bins=50, edgecolor='black')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Word Count')
plt.axvline(train_df['word_count'].median(), color='red', linestyle='--', label=f'Median: {train_df["word_count"].median():.0f}')
plt.legend()

plt.tight_layout()
plt.savefig('text_length_distribution.png', dpi=300, bbox_inches='tight')
print("\nSaved: text_length_distribution.png")

# Primary Category Analysis
print("\n" + "="*80)
print("PRIMARY CATEGORY DISTRIBUTION")
print("="*80)
primary_counts = train_df['primary_category'].value_counts()
print(primary_counts)
print(f"\nUnique primary categories: {train_df['primary_category'].nunique()}")

plt.figure(figsize=(12, 6))
primary_counts.head(15).plot(kind='barh')
plt.xlabel('Count')
plt.title('Top 15 Primary Categories')
plt.tight_layout()
plt.savefig('primary_category_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: primary_category_distribution.png")

# Secondary Category Analysis
print("\n" + "="*80)
print("SECONDARY CATEGORY DISTRIBUTION")
print("="*80)
secondary_counts = train_df['secondary_category'].value_counts()
print(secondary_counts.head(20))
print(f"\nUnique secondary categories: {train_df['secondary_category'].nunique()}")

plt.figure(figsize=(12, 8))
secondary_counts.head(20).plot(kind='barh')
plt.xlabel('Count')
plt.title('Top 20 Secondary Categories')
plt.tight_layout()
plt.savefig('secondary_category_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: secondary_category_distribution.png")

# Severity Analysis
print("\n" + "="*80)
print("SEVERITY DISTRIBUTION")
print("="*80)
severity_counts = train_df['severity'].value_counts().sort_index()
print(severity_counts)
print(f"\nSeverity range: {train_df['severity'].min()} to {train_df['severity'].max()}")

plt.figure(figsize=(8, 5))
severity_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Severity Level')
plt.ylabel('Count')
plt.title('Severity Distribution')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('severity_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: severity_distribution.png")

# Check for class imbalance
print("\n" + "="*80)
print("CLASS IMBALANCE CHECK")
print("="*80)
print(f"Primary category - Most common: {primary_counts.iloc[0]} ({primary_counts.iloc[0]/len(train_df)*100:.1f}%)")
print(f"Primary category - Least common: {primary_counts.iloc[-1]} ({primary_counts.iloc[-1]/len(train_df)*100:.1f}%)")
print(f"Imbalance ratio: {primary_counts.iloc[0]/primary_counts.iloc[-1]:.1f}x")

print(f"\nSecondary category - Most common: {secondary_counts.iloc[0]} ({secondary_counts.iloc[0]/len(train_df)*100:.1f}%)")
print(f"Secondary category - Least common: {secondary_counts.iloc[-1]} ({secondary_counts.iloc[-1]/len(train_df)*100:.1f}%)")
print(f"Imbalance ratio: {secondary_counts.iloc[0]/secondary_counts.iloc[-1]:.1f}x")

# Sample complaints by severity
print("\n" + "="*80)
print("SAMPLE COMPLAINTS BY SEVERITY")
print("="*80)
for severity in sorted(train_df['severity'].unique()):
    print(f"\n--- SEVERITY {severity} ---")
    sample = train_df[train_df['severity'] == severity].iloc[0]
    print(f"Primary: {sample['primary_category']}")
    print(f"Secondary: {sample['secondary_category']}")
    print(f"Text: {sample['complaint_text'][:200]}...")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Primary categories: {train_df['primary_category'].nunique()}")
print(f"Secondary categories: {train_df['secondary_category'].nunique()}")
print(f"Severity levels: {sorted(train_df['severity'].unique())}")