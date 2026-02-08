import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm
import pickle
from config import config

# Set random seeds
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

print("="*80)
print(f"TRAINING MULTI-TASK COMPLAINT CLASSIFIER")
print("="*80)
print(f"Model: {config.MODEL_NAME}")
print(f"Device: {config.DEVICE}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Learning rate: {config.LEARNING_RATE}")
print(f"Epochs: {config.NUM_EPOCHS}")

# ============================================================================
# DATASET CLASS
# ============================================================================

class ComplaintDataset(Dataset):
    def __init__(self, texts, primary_labels, secondary_labels, severity_labels, tokenizer, max_length):
        self.texts = texts
        self.primary_labels = primary_labels
        self.secondary_labels = secondary_labels
        self.severity_labels = severity_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'primary': torch.tensor(self.primary_labels[idx], dtype=torch.long),
            'secondary': torch.tensor(self.secondary_labels[idx], dtype=torch.long),
            'severity': torch.tensor(self.severity_labels[idx], dtype=torch.long)
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

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

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, primary_weight, secondary_weight, severity_weight):
    model.train()
    total_loss = 0
    primary_correct = 0
    secondary_correct = 0
    severity_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        primary_labels = batch['primary'].to(device)
        secondary_labels = batch['secondary'].to(device)
        severity_labels = batch['severity'].to(device)
        
        # Forward pass
        primary_logits, secondary_logits, severity_logits = model(input_ids, attention_mask)
        
        # Compute losses (all classification)
        primary_loss = nn.CrossEntropyLoss()(primary_logits, primary_labels)
        secondary_loss = nn.CrossEntropyLoss()(secondary_logits, secondary_labels)
        severity_loss = nn.CrossEntropyLoss()(severity_logits, severity_labels)
        
        # Weighted combination
        loss = primary_weight * primary_loss + secondary_weight * secondary_loss + severity_weight * severity_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        primary_preds = torch.argmax(primary_logits, dim=1)
        secondary_preds = torch.argmax(secondary_logits, dim=1)
        severity_preds = torch.argmax(severity_logits, dim=1)
        
        primary_correct += (primary_preds == primary_labels).sum().item()
        secondary_correct += (secondary_preds == secondary_labels).sum().item()
        severity_correct += (severity_preds == severity_labels).sum().item()
        
        total_loss += loss.item()
        total_samples += input_ids.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'primary_acc': primary_correct / total_samples,
            'secondary_acc': secondary_correct / total_samples
        })
    
    avg_loss = total_loss / len(dataloader)
    primary_acc = primary_correct / total_samples
    secondary_acc = secondary_correct / total_samples
    severity_acc = severity_correct / total_samples
    
    return avg_loss, primary_acc, secondary_acc, severity_acc

def evaluate(model, dataloader, device, primary_weight, secondary_weight, severity_weight):
    model.eval()
    total_loss = 0
    
    all_primary_preds = []
    all_primary_labels = []
    all_secondary_preds = []
    all_secondary_labels = []
    all_severity_preds = []
    all_severity_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            primary_labels = batch['primary'].to(device)
            secondary_labels = batch['secondary'].to(device)
            severity_labels = batch['severity'].to(device)
            
            # Forward pass
            primary_logits, secondary_logits, severity_logits = model(input_ids, attention_mask)
            
            # Compute losses
            primary_loss = nn.CrossEntropyLoss()(primary_logits, primary_labels)
            secondary_loss = nn.CrossEntropyLoss()(secondary_logits, secondary_labels)
            severity_loss = nn.CrossEntropyLoss()(severity_logits, severity_labels)
            
            loss = primary_weight * primary_loss + secondary_weight * secondary_loss + severity_weight * severity_loss
            total_loss += loss.item()
            
            # Get predictions
            primary_preds = torch.argmax(primary_logits, dim=1)
            secondary_preds = torch.argmax(secondary_logits, dim=1)
            severity_preds = torch.argmax(severity_logits, dim=1)
            
            # Store for metrics
            all_primary_preds.extend(primary_preds.cpu().numpy())
            all_primary_labels.extend(primary_labels.cpu().numpy())
            all_secondary_preds.extend(secondary_preds.cpu().numpy())
            all_secondary_labels.extend(secondary_labels.cpu().numpy())
            all_severity_preds.extend(severity_preds.cpu().numpy())
            all_severity_labels.extend(severity_labels.cpu().numpy())
    
    # Calculate metrics
    primary_acc = accuracy_score(all_primary_labels, all_primary_preds)
    secondary_acc = accuracy_score(all_secondary_labels, all_secondary_preds)
    severity_acc = accuracy_score(all_severity_labels, all_severity_preds)
    
    # R² for severity (convert back to 1-5 scale)
    severity_preds_original = np.array(all_severity_preds) + 1
    severity_labels_original = np.array(all_severity_labels) + 1
    severity_r2 = r2_score(severity_labels_original, severity_preds_original)
    
    # Combined score
    combined_score = (
        primary_weight * primary_acc +
        secondary_weight * secondary_acc +
        severity_weight * severity_r2
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, primary_acc, secondary_acc, severity_acc, severity_r2, combined_score

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_df = pd.read_csv(config.TRAIN_PATH)
    
    # Split data
    train_data, val_data = train_test_split(
        train_df,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=train_df['primary_category']
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Encode labels
    print("\nEncoding labels...")
    primary_encoder = LabelEncoder()
    secondary_encoder = LabelEncoder()
    
    train_data = train_data.copy()
    val_data = val_data.copy()
    
    train_data['primary_encoded'] = primary_encoder.fit_transform(train_data['primary_category'])
    train_data['secondary_encoded'] = secondary_encoder.fit_transform(train_data['secondary_category'])
    train_data['severity_encoded'] = train_data['severity'] - 1  # 1-5 → 0-4
    
    val_data['primary_encoded'] = primary_encoder.transform(val_data['primary_category'])
    val_data['secondary_encoded'] = secondary_encoder.transform(val_data['secondary_category'])
    val_data['severity_encoded'] = val_data['severity'] - 1
    
    num_primary = len(primary_encoder.classes_)
    num_secondary = len(secondary_encoder.classes_)
    
    print(f"Primary categories: {num_primary}")
    print(f"Secondary categories: {num_secondary}")
    print(f"Severity levels: 5")
    
    # Save encoders
    with open('models/primary_encoder.pkl', 'wb') as f:
        pickle.dump(primary_encoder, f)
    with open('models/secondary_encoder.pkl', 'wb') as f:
        pickle.dump(secondary_encoder, f)
    print("\nSaved label encoders")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ComplaintDataset(
        texts=train_data['complaint_text'].values,
        primary_labels=train_data['primary_encoded'].values,
        secondary_labels=train_data['secondary_encoded'].values,
        severity_labels=train_data['severity_encoded'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = ComplaintDataset(
        texts=val_data['complaint_text'].values,
        primary_labels=val_data['primary_encoded'].values,
        secondary_labels=val_data['secondary_encoded'].values,
        severity_labels=val_data['severity_encoded'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    model = MultiTaskComplaintClassifier(
        model_name=config.MODEL_NAME,
        num_primary=num_primary,
        num_secondary=num_secondary,
        dropout=config.DROPOUT
    )
    model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {config.WARMUP_STEPS}")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_combined_score = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_primary_acc': [],
        'val_secondary_acc': [],
        'val_severity_r2': [],
        'val_combined_score': []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_primary_acc, train_secondary_acc, train_severity_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE,
            config.PRIMARY_WEIGHT, config.SECONDARY_WEIGHT, config.SEVERITY_WEIGHT
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Primary Acc: {train_primary_acc:.4f}")
        print(f"Train Secondary Acc: {train_secondary_acc:.4f}")
        print(f"Train Severity Acc: {train_severity_acc:.4f}")
        
        # Evaluate
        val_loss, val_primary_acc, val_secondary_acc, val_severity_acc, val_severity_r2, val_combined = evaluate(
            model, val_loader, config.DEVICE,
            config.PRIMARY_WEIGHT, config.SECONDARY_WEIGHT, config.SEVERITY_WEIGHT
        )
        
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Primary Acc: {val_primary_acc:.4f}")
        print(f"Validation Secondary Acc: {val_secondary_acc:.4f}")
        print(f"Validation Severity Acc: {val_severity_acc:.4f}")
        print(f"Validation Severity R²: {val_severity_r2:.4f}")
        print(f"\n{'='*40}")
        print(f"COMBINED SCORE: {val_combined:.4f}")
        print(f"{'='*40}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_primary_acc'].append(val_primary_acc)
        history['val_secondary_acc'].append(val_secondary_acc)
        history['val_severity_r2'].append(val_severity_r2)
        history['val_combined_score'].append(val_combined)
        
        # Save best model
        if val_combined > best_combined_score:
            best_combined_score = val_combined
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_combined_score,
                'config': config
            }, config.MODEL_SAVE_PATH)
            print(f"\n✓ New best model saved! Score: {best_combined_score:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Combined Score: {best_combined_score:.4f}")
    
    # Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("\nTraining history saved")

if __name__ == '__main__':
    main()