import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import logging
import datetime
import sys
from feature_engineering import FeatureExtractor, ensure_nltk_resources
from dataset_loader import load_phishing_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PhishingModelTrainer:
    def __init__(self):
        # Ensure NLTK resources are downloaded
        ensure_nltk_resources()
        
        self.feature_extractor = FeatureExtractor()
        self.model_dir = "models"
        self.feedback_dir = "feedback"
        
        # Create necessary directories
        for directory in [self.model_dir, self.feedback_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def preprocess_data(self, df):
        """
        Extract features from each email in the dataset
        
        Args:
            df: Pandas DataFrame with 'text' and 'label' columns
            
        Returns:
            Pandas DataFrame with extracted features and labels
        """
        logger.info("Preprocessing dataset...")
        
        # Extract features from each email
        features_list = []
        for idx, row in enumerate(df.iterrows()):
            if idx % 100 == 0:
                logger.info(f"Processing email {idx}/{len(df)}")
            
            try:
                features = self.feature_extractor.extract_features(row[1]['text'])
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error processing email {idx}: {e}")
                # Use a default set of features
                default_features = {
                    'cleaned_text': '',
                    'url_count': 0,
                    'has_url': 0,
                    'email_count': 0,
                    'has_email': 0,
                    'word_count': 0,
                    'char_count': 0,
                    'avg_word_length': 0,
                    'contains_number': 0,
                    'spelling_errors': 0,
                    'grammar_score': 0.5,
                    'sentiment': 0,
                    'suspicious_word_ratio': 0,
                    'has_html': 0,
                    'has_script': 0,
                    'has_form': 0,
                    'has_javascript': 0,
                    'capital_run_length': 0,
                    'number_of_hyperlinks': 0
                }
                features_list.append(default_features)
        
        # Convert to dataframe
        features_df = pd.DataFrame(features_list)
        
        # Add labels
        features_df['label'] = df['label'].values
        
        return features_df
    
    def train_model(self, save_model=True):
        """
        Train a phishing detection model
        
        Args:
            save_model: Whether to save the trained model to disk
            
        Returns:
            trained model, performance report
        """
        logger.info("Starting model training...")
        
        # Load dataset
        train_data, test_data = load_phishing_dataset()
        
        # Preprocess data
        processed_train_df = self.preprocess_data(train_data)
        processed_test_df = self.preprocess_data(test_data)
        
        # Prepare features and target
        X_train = processed_train_df.drop('label', axis=1)
        y_train = processed_train_df['label']
        X_test = processed_test_df.drop('label', axis=1)
        y_test = processed_test_df['label']
        
        # Select only numeric features
        num_features = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[num_features]
        X_test = X_test[num_features]
        
        # Train Random Forest classifier
        logger.info("Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model performance:\n{classification_report(y_test, y_pred)}")
        logger.info(f"Confusion matrix:\n{conf_matrix}")
        
        # Save the model
        if save_model:
            model_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"phishing_model_{model_timestamp}.pkl"
            model_path = os.path.join(self.model_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': list(X_train.columns),
                    'performance': report,
                    'confusion_matrix': conf_matrix,
                    'training_date': datetime.datetime.now().isoformat(),
                    'model_type': 'RandomForest'
                }, f)
            
            logger.info(f"Model saved to {model_path}")
            
            # Save feature extractor
            feature_extractor_path = os.path.join(self.model_dir, "feature_extractor.pkl")
            with open(feature_extractor_path, 'wb') as f:
                pickle.dump(self.feature_extractor, f)
            
            logger.info(f"Feature extractor saved to {feature_extractor_path}")
        
        return model, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train phishing email detection model')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    args = parser.parse_args()
    
    try:
        # First, ensure NLTK resources are downloaded
        ensure_nltk_resources()
        
        # Create trainer and train model
        trainer = PhishingModelTrainer()
        model, report = trainer.train_model(save_model=not args.no_save)
        
        # Print performance metrics
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        print("\n========== MODEL PERFORMANCE ==========")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    
    except Exception as e:
        print(f"\n⚠️ ERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Run the setup script first: python setup.py")
        print("2. Make sure you have an internet connection for downloading the dataset")
        print("3. Try running these commands to manually install NLTK data:")
        print("   >>> import nltk")
        print("   >>> nltk.download('punkt')")
        print("   >>> nltk.download('stopwords')")
        print("4. Check for proper permissions in your directory")
        print("5. Ensure you have the required packages by running: pip install -r requirements.txt")
        sys.exit(1)
