# phishing_detector.py
import sys
import pickle
import os
import pandas as pd
import datetime
import argparse
import logging
import colorama
from colorama import Fore, Style
# At the top of phishing_detector.py add:
from feature_engineering import ensure_nltk_resources

# Make sure NLTK resources are available
ensure_nltk_resources()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phishing_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize colorama
colorama.init()

class PhishingDetector:
    def __init__(self):
        self.model_dir = "models"
        self.feedback_dir = "feedback"
        self.feature_extractor = None
        self.model = None
        self.model_data = None
        self.features = None
        
        # Create directories if they don't exist
        for directory in [self.model_dir, self.feedback_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_latest_model(self):
        """
        Load the latest trained model from the models directory
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith("phishing_model_") and f.endswith(".pkl")]
        
        if not model_files:
            logger.error("No model files found. Please train a model first.")
            return False
        
        # Sort by timestamp (newest first)
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        model_path = os.path.join(self.model_dir, latest_model)
        
        # Load feature extractor
        feature_extractor_path = os.path.join(self.model_dir, "feature_extractor.pkl")
        if not os.path.exists(feature_extractor_path):
            logger.error("Feature extractor not found. Please train a model first.")
            return False
        
        try:
            with open(feature_extractor_path, 'rb') as f:
                self.feature_extractor = pickle.load(f)
            
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
                self.model = self.model_data['model']
                self.features = self.model_data['features']
            
            logger.info(f"Loaded model: {latest_model}")
            logger.info(f"Model performance: Accuracy={self.model_data['performance']['accuracy']:.4f}, "
                      f"F1={self.model_data['performance']['weighted avg']['f1-score']:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def detect_phishing(self, email_content):
        """
        Detect if an email is phishing
        
        Args:
            email_content: Email text content
            
        Returns:
            dict: Detection results with prediction, confidence, and explanation
        """
        if self.model is None or self.feature_extractor is None:
            if not self.load_latest_model():
                return None
        
        # Extract features
        features = self.feature_extractor.extract_features(email_content)
        
        # Convert to DataFrame and select only features used by the model
        features_df = pd.DataFrame([features])
        selected_features = [f for f in self.features if f in features_df.columns]
        
        # If any features are missing, add them with default values
        for feature in self.features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Make prediction
        prediction = self.model.predict(features_df[self.features])[0]
        probabilities = self.model.predict_proba(features_df[self.features])[0]
        
        # Get confidence score
        confidence = probabilities[prediction]
        
        # Generate explanation
        explanation = self.generate_explanation(features, prediction, probabilities)
        
        # Return results
        return {
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': probabilities,
            'explanation': explanation,
            'features': features
        }
    
    def generate_explanation(self, features, prediction, probabilities):
        """
        Generate a human-readable explanation for the prediction
        
        Args:
            features: Extracted email features
            prediction: Model prediction (0 = safe, 1 = phishing)
            probabilities: Prediction probabilities
            
        Returns:
            list: List of explanation points
        """
        explanation = []
        
        if prediction == 1:  # Phishing
            # URLs are strong indicators of phishing
            if features['url_count'] > 0:
                explanation.append(f"Contains {features['url_count']} URLs")
            
            # Check for suspicious content
            if features['suspicious_word_ratio'] > 0.05:
                explanation.append("Contains suspicious wording commonly found in phishing emails")
            
            # Check for HTML/JS
            if features['has_html'] and features['has_form']:
                explanation.append("Contains HTML forms (often used to steal credentials)")
            
            if features['has_javascript']:
                explanation.append("Contains JavaScript (could be used for malicious purposes)")
            
            # Check sentiment
            if features['sentiment'] < -0.2:
                explanation.append("Uses negative or urgent tone")
            
            # Check structure
            if features['capital_run_length'] > 3:
                explanation.append("Excessive use of CAPITAL LETTERS")
            
            # Generic fallback explanation
            if not explanation:
                explanation.append(f"Detected as phishing with {probabilities[1]*100:.1f}% confidence")
        else:  # Safe
            explanation.append("No significant phishing indicators detected")
            
            if features['url_count'] > 0 and features['suspicious_word_ratio'] < 0.03:
                explanation.append("URLs in email appear legitimate")
            
            if features['suspicious_word_ratio'] < 0.03:
                explanation.append("Does not contain typical phishing vocabulary")
            
            # Generic confidence explanation
            explanation.append(f"Detected as safe with {probabilities[0]*100:.1f}% confidence")
        
        return explanation
    
    def save_feedback(self, email_content, prediction, user_feedback):
        """
        Save user feedback for model improvement
        
        Args:
            email_content: Email text
            prediction: Model prediction
            user_feedback: User feedback (0 = safe, 1 = phishing)
            
        Returns:
            bool: True if feedback saved successfully
        """
        feedback_file = os.path.join(
            self.feedback_dir, 
            f"feedback_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        # Create feedback entry
        feedback_data = {
            'text': [email_content],
            'label': [int(user_feedback)],
            'model_prediction': [int(prediction)],
            'timestamp': [datetime.datetime.now().isoformat()]
        }
        
        feedback_df = pd.DataFrame(feedback_data)
        
        # Append or create feedback file
        try:
            if os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_file, index=False)
            
            logger.info(f"Saved user feedback to {feedback_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Phishing Email Detection CLI')
    parser.add_argument('--file', help='Path to email file to analyze')
    
    args = parser.parse_args()
    
    detector = PhishingDetector()
    
    print(f"{Fore.CYAN}=================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}   PHISHING EMAIL DETECTOR v1.0   {Style.RESET_ALL}")
    print(f"{Fore.CYAN}=================================================={Style.RESET_ALL}")
    
    # Load the model
    if not detector.load_latest_model():
        print(f"{Fore.RED}Error: Could not load model. Please train a model first.{Style.RESET_ALL}")
        print(f"Run: python train_model.py to train a model.")
        return
    
    # Get email content either from file or stdin
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read().strip()
        except Exception as e:
            print(f"{Fore.RED}Error reading file: {str(e)}{Style.RESET_ALL}")
            return
    else:
        print(f"{Fore.YELLOW}Paste the email content below and press Ctrl+D (Unix) or Ctrl+Z+Enter (Windows) when finished:{Style.RESET_ALL}")
        try:
            email_content = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return
    
    if not email_content:
        print(f"{Fore.RED}Error: No email content provided.{Style.RESET_ALL}")
        return
    
    # Analyze the email
    print(f"{Fore.YELLOW}Analyzing email...{Style.RESET_ALL}")
    result = detector.detect_phishing(email_content)
    
    if result is None:
        print(f"{Fore.RED}Error analyzing email. Please ensure the model is properly trained.{Style.RESET_ALL}")
        return
    
    # Display results
    print("\n")
    print(f"{Fore.CYAN}============== ANALYSIS RESULT =============={Style.RESET_ALL}")
    
    if result['prediction'] == 1:
        print(f"{Fore.RED}VERDICT: ⚠️  PHISHING EMAIL DETECTED  ⚠️{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Confidence: {result['confidence']*100:.2f}%{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}VERDICT: ✓ LEGITIMATE EMAIL ✓{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Confidence: {result['confidence']*100:.2f}%{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Explanation:{Style.RESET_ALL}")
    for point in result['explanation']:
        if result['prediction'] == 1:
            print(f" {Fore.RED}•{Style.RESET_ALL} {point}")
        else:
            print(f" {Fore.GREEN}•{Style.RESET_ALL} {point}")
    
    # Ask for feedback
    print(f"\n{Fore.CYAN}Was this analysis correct? (y/n):{Style.RESET_ALL}")
    try:
        feedback = input().strip().lower()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return
    
    if feedback in ('y', 'yes'):
        print(f"{Fore.GREEN}Thank you for your feedback!{Style.RESET_ALL}")
        detector.save_feedback(email_content, result['prediction'], result['prediction'])
    elif feedback in ('n', 'no'):
        print(f"{Fore.YELLOW}Could you tell us the correct classification? (1 for phishing, 0 for legitimate):{Style.RESET_ALL}")
        try:
            correct_label = input().strip()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return
            
        if correct_label in ('0', '1'):
            detector.save_feedback(email_content, result['prediction'], int(correct_label))
            print(f"{Fore.GREEN}Thank you for your feedback! This will help improve future model updates.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Invalid input. Feedback not saved.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No feedback provided.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
