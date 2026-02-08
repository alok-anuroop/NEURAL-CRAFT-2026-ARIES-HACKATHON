import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from config import config

print("="*80)
print("BASELINE MODEL: TF-IDF + Logistic Regression")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv(config.TRAIN_PATH)

# Split data
train_data, val_data = train_test_split(
    train_df, 
    test_size=config.VALIDATION_SPLIT, 
    random_state=config.RANDOM_SEED,
    stratify=train_df['primary_category']  # Stratify by primary category
)

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Encode labels
primary_encoder = LabelEncoder()
secondary_encoder = LabelEncoder()

train_data['primary_encoded'] = primary_encoder.fit_transform(train_data['primary_category'])
train_data['secondary_encoded'] = secondary_encoder.fit_transform(train_data['secondary_category'])
train_data['severity_encoded'] = train_data['severity'] - 1  # Convert 1-5 to 0-4

val_data['primary_encoded'] = primary_encoder.transform(val_data['primary_category'])
val_data['secondary_encoded'] = secondary_encoder.transform(val_data['secondary_category'])
val_data['severity_encoded'] = val_data['severity'] - 1

# TF-IDF vectorization
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(train_data['complaint_text'])
X_val = vectorizer.transform(val_data['complaint_text'])

print(f"TF-IDF feature shape: {X_train.shape}")

# Train separate models for each task
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

# Primary Category Model
print("\n1. Training Primary Category classifier...")
primary_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
primary_model.fit(X_train, train_data['primary_encoded'])
primary_pred = primary_model.predict(X_val)
primary_acc = accuracy_score(val_data['primary_encoded'], primary_pred)
print(f"   Primary Accuracy: {primary_acc:.4f}")

# Secondary Category Model
print("\n2. Training Secondary Category classifier...")
secondary_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
secondary_model.fit(X_train, train_data['secondary_encoded'])
secondary_pred = secondary_model.predict(X_val)
secondary_acc = accuracy_score(val_data['secondary_encoded'], secondary_pred)
print(f"   Secondary Accuracy: {secondary_acc:.4f}")

# Severity Model
print("\n3. Training Severity classifier...")
severity_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
severity_model.fit(X_train, train_data['severity_encoded'])
severity_pred = severity_model.predict(X_val)
severity_acc = accuracy_score(val_data['severity_encoded'], severity_pred)
# For R² calculation, add 1 back to get 1-5 range
severity_r2 = r2_score(val_data['severity'], severity_pred + 1)
print(f"   Severity Accuracy: {severity_acc:.4f}")
print(f"   Severity R²: {severity_r2:.4f}")

# Calculate combined score
print("\n" + "="*80)
print("BASELINE RESULTS")
print("="*80)
combined_score = (
    config.PRIMARY_WEIGHT * primary_acc +
    config.SECONDARY_WEIGHT * secondary_acc +
    config.SEVERITY_WEIGHT * severity_r2
)
print(f"\nPrimary Accuracy:    {primary_acc:.4f} (weight: {config.PRIMARY_WEIGHT})")
print(f"Secondary Accuracy:  {secondary_acc:.4f} (weight: {config.SECONDARY_WEIGHT})")
print(f"Severity R²:         {severity_r2:.4f} (weight: {config.SEVERITY_WEIGHT})")
print(f"\n{'='*40}")
print(f"COMBINED SCORE:      {combined_score:.4f}")
print(f"{'='*40}")

print("\nBaseline established! Now let's beat this with deep learning 🚀")