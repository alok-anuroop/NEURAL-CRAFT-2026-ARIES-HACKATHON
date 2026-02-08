import torch

class Config:
    # Paths
    TRAIN_PATH = 'data/train_complaints.csv'
    TEST_PATH = 'data/test_complaints.csv'
    MODEL_SAVE_PATH = 'models/best_model.pt'
    SUBMISSION_PATH = 'submissions/submission.csv'
    
    # Model
    MODEL_NAME = 'bert-base-uncased'  # or 'distilbert-base-uncased' for faster training
    MAX_LENGTH = 256  # Maximum token length
    
    # Training
    BATCH_SIZE = 16  # Reduce to 8 if GPU memory issues
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Multi-task loss weights (matching evaluation metric)
    PRIMARY_WEIGHT = 0.3
    SECONDARY_WEIGHT = 0.4
    SEVERITY_WEIGHT = 0.3
    
    # Other
    RANDOM_SEED = 42
    DROPOUT = 0.2
    VALIDATION_SPLIT = 0.15
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __repr__(self):
        return f"Config(model={self.MODEL_NAME}, batch_size={self.BATCH_SIZE}, lr={self.LEARNING_RATE})"

config = Config()
print(config)
print(f"Using device: {config.DEVICE}")
