import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from config import config
import torch.nn as nn
from transformers import AutoModel

# Define model class directly here
class MultiTaskComplaintClassifier(nn.Module):
    def __init__(self, model_name, num_primary, num_secondary, dropout=0.2):
        super().__init__()
        
        # Shared encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Task-specific heads (all classification)
        self.primary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_primary)
        )
        
        self.secondary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_secondary)
        )
        
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 5)  # 5 severity classes (1-5)
        )
    
    def forward(self, input_ids, attention_mask):
        # Shared encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Task predictions (all return logits)
        primary_logits = self.primary_head(pooled)
        secondary_logits = self.secondary_head(pooled)
        severity_logits = self.severity_head(pooled)
        
        return primary_logits, secondary_logits, severity_logits

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

if __name__ == '__main__':
    import os
    os.makedirs('submissions', exist_ok=True)
    print("="*80)
    print("GENERATING PREDICTIONS FOR TEST SET")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(config.TEST_PATH)
    print(f"Test samples: {len(test_df)}")
    
    # Load encoders
    print("\nLoading label encoders...")
    with open('models/primary_encoder.pkl', 'rb') as f:
        primary_encoder = pickle.load(f)
    with open('models/secondary_encoder.pkl', 'rb') as f:
        secondary_encoder = pickle.load(f)
    
    num_primary = len(primary_encoder.classes_)
    num_secondary = len(secondary_encoder.classes_)
    
    print(f"Primary categories: {num_primary}")
    print(f"Secondary categories: {num_secondary}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create dataset
    print("\nCreating test dataset...")
    test_dataset = TestDataset(
        texts=test_df['complaint_text'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Changed to 0 for Windows
        pin_memory=False  # Changed to False since using CPU
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model = MultiTaskComplaintClassifier(
        model_name=config.MODEL_NAME,
        num_primary=num_primary,
        num_secondary=num_secondary,
        dropout=config.DROPOUT
    )
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with score {checkpoint['best_score']:.4f}")
    
    # Generate predictions
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    all_primary_preds = []
    all_secondary_preds = []
    all_severity_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            # Forward pass
            primary_logits, secondary_logits, severity_logits = model(input_ids, attention_mask)
            
            # Get predictions
            primary_preds = torch.argmax(primary_logits, dim=1).cpu().numpy()
            secondary_preds = torch.argmax(secondary_logits, dim=1).cpu().numpy()
            severity_preds = torch.argmax(severity_logits, dim=1).cpu().numpy()
            
            all_primary_preds.extend(primary_preds)
            all_secondary_preds.extend(secondary_preds)
            all_severity_preds.extend(severity_preds)
    
    # Convert predictions back to original labels
    print("\nDecoding predictions...")
    primary_decoded = primary_encoder.inverse_transform(all_primary_preds)
    secondary_decoded = secondary_encoder.inverse_transform(all_secondary_preds)
    severity_decoded = np.array(all_severity_preds) + 1  # 0-4 → 1-5
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'complaint_id': test_df['complaint_id'],
        'primary_category': primary_decoded,
        'secondary_category': secondary_decoded,
        'severity': severity_decoded
    })
    
    # Save submission
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    
    print("\n" + "="*80)
    print("SUBMISSION CREATED")
    print("="*80)
    print(f"Saved to: {config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Display sample predictions
    print("\nSample predictions:")
    print(submission_df.head(10))
    
    # Check submission format
    print("\n" + "="*80)
    print("SUBMISSION VALIDATION")
    print("="*80)
    print(f"✓ Column names: {list(submission_df.columns)}")
    print(f"✓ Number of rows: {len(submission_df)}")
    print(f"✓ Null values: {submission_df.isnull().sum().sum()}")
    print(f"✓ Unique complaint IDs: {submission_df['complaint_id'].nunique()}")
    print(f"✓ Severity range: {submission_df['severity'].min()} to {submission_df['severity'].max()}")
    print(f"✓ Primary categories: {submission_df['primary_category'].nunique()}")
    print(f"✓ Secondary categories: {submission_df['secondary_category'].nunique()}")
    
    print("\nReady for submission! 🚀")