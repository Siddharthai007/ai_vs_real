import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class CustomImageDataset(Dataset):
    """Custom dataset for loading real and AI-generated images"""
    
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            logger.warning(f"Error loading {self.image_paths[idx]}: {e}")
            return {
                'pixel_values': torch.zeros((3, 224, 224)),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

def prepare_data(data_dir='data'):
    """Prepare training data from directory structure"""
    image_paths = []
    labels = []
    
    # Load real images (label: 0)
    real_dir = os.path.join(data_dir, 'real')
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(real_dir, filename))
                labels.append(0)
    
    # Load fake images (label: 1)
    fake_dir = os.path.join(data_dir, 'fake')
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(fake_dir, filename))
                labels.append(1)
    
    logger.info(f"Total images found: {len(image_paths)}")
    logger.info(f"Real images: {labels.count(0)}, Fake images: {labels.count(1)}")
    
    # Train-test split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return (train_paths, train_labels), (val_paths, val_labels)

def compute_metrics(eval_preds):
    """Compute accuracy metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    logits, labels = eval_preds
    predictions = logits.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

def train_model():
    """Train Vision Transformer model"""
    
    # Initialize processor and model
    logger.info("Loading Vision Transformer model...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1}
    )
    
    # Prepare data
    logger.info("Preparing dataset...")
    (train_paths, train_labels), (val_paths, val_labels) = prepare_data()
    
    # Create datasets
    train_dataset = CustomImageDataset(train_paths, train_labels, processor)
    val_dataset = CustomImageDataset(val_paths, val_labels, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./model_output',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        remove_unused_columns=False,
    )
    
    # Trainer
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model('./model_dir')
    processor.save_pretrained('./model_dir')
    
    logger.info("Model training completed and saved to ./model_dir")

if __name__ == '__main__':
    train_model()
