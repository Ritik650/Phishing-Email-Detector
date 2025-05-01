from datasets import load_dataset
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)

def load_phishing_dataset(split_ratio=0.2, random_state=42):
    """
    Load the phishing email dataset from Hugging Face
    
    Args:
        split_ratio: Ratio of test data (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, test_df: Pandas DataFrames containing the training and test data
    """
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("zefang-liu/phishing-email-dataset")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Show available columns
        print(f"Available columns in dataset: {df.columns.tolist()}")
        
        # Rename columns to match our expected format - adapt these based on actual column names
        if 'Email Text' in df.columns and 'Email Type' in df.columns:
            df = df.rename(columns={
                'Email Text': 'text',
                'Email Type': 'label'
            })
        
        # Convert labels to binary (1 for Phishing, 0 for Safe)
        if df['label'].dtype == object:
            df['label'] = df['label'].apply(lambda x: 1 if 'phish' in str(x).lower() else 0)
        
        # Remove any rows with null values
        df = df.dropna(subset=['text', 'label'])
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split into train and test sets
        test_size = int(len(df) * split_ratio)
        train_df = df[test_size:]
        test_df = df[:test_size]
        
        print(f"Loaded dataset with {len(df)} emails")
        print(f"Training set: {len(train_df)} emails")
        print(f"Test set: {len(test_df)} emails")
        print(f"Phishing emails in training set: {train_df['label'].sum()} ({train_df['label'].mean()*100:.2f}%)")
        print(f"Phishing emails in test set: {test_df['label'].sum()} ({test_df['label'].mean()*100:.2f}%)")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
