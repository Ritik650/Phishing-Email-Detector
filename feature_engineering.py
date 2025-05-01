import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import logging


logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
def ensure_nltk_resources():
    """Ensure NLTK resources are downloaded"""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource, quiet=True)

# Download resources when module is imported
ensure_nltk_resources()

class FeatureExtractor:
    def __init__(self):
        # Ensure resources are downloaded before initializing
        ensure_nltk_resources()
        
        self.stop_words = set(stopwords.words('english'))
        # Load suspicious words list
        self.suspicious_words = {'urgent', 'verify', 'account', 'suspend', 'bank', 
                              'confirm', 'update', 'login', 'click', 'password',
                              'access', 'credit', 'free', 'link', 'alert', 'secure',
                              'record', 'document', 'statement', 'identity', 'confirm',
                              'restricted', 'remove', 'ssn', 'social security', 'payment'}
        
    def extract_features(self, email_content):
        """
        Extract features from email content
        
        Args:
            email_content: Raw email text
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        if not isinstance(email_content, str):
            email_content = str(email_content)
        
        # Basic text cleaning
        cleaned_text = self._clean_text(email_content)
        features['cleaned_text'] = cleaned_text
        
        # URL features
        urls = self._extract_urls(email_content)
        features['url_count'] = len(urls)
        features['has_url'] = 1 if len(urls) > 0 else 0
        
        # Email features
        emails = self._extract_emails(email_content)
        features['email_count'] = len(emails)
        features['has_email'] = 1 if len(emails) > 0 else 0
        
        # Linguistic features
        features['word_count'] = len(cleaned_text.split())
        features['char_count'] = len(cleaned_text)
        features['avg_word_length'] = self._avg_word_length(cleaned_text)
        features['contains_number'] = 1 if re.search(r'\d', cleaned_text) else 0
        features['spelling_errors'] = self._count_spelling_errors(cleaned_text)
        features['grammar_score'] = self._grammar_check(cleaned_text)
        features['sentiment'] = self._sentiment_analysis(cleaned_text)
        features['suspicious_word_ratio'] = self._suspicious_word_ratio(cleaned_text)
        
        # Structure features
        features['has_html'] = 1 if '<html' in email_content.lower() else 0
        features['has_script'] = 1 if '<script' in email_content.lower() else 0
        features['has_form'] = 1 if '<form' in email_content.lower() else 0
        features['has_javascript'] = 1 if 'javascript' in email_content.lower() else 0
        features['capital_run_length'] = self._capital_run_length(email_content)
        features['number_of_hyperlinks'] = email_content.lower().count('href=')
        
        return features
    
    def _clean_text(self, text):
        """Clean the text by removing HTML tags, special characters, etc."""
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            # Split into words and remove stopwords
            words = [word for word in text.split() if word not in self.stop_words]
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Error in text cleaning: {e}")
            return text
    
    def _extract_urls(self, text):
        """Extract URLs from text"""
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        return url_pattern.findall(text)
    
    def _extract_emails(self, text):
        """Extract email addresses from text"""
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        return email_pattern.findall(text)
    
    def _avg_word_length(self, text):
        """Calculate average word length"""
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def _count_spelling_errors(self, text):
        """Count spelling errors using TextBlob"""
        try:
            words = text.split()
            if not words:
                return 0
                
            textblob = TextBlob(text)
            misspelled = 0
            for word in textblob.words:
                if word.lower() not in self.stop_words and len(word) > 3:
                    suggestions = textblob.correct()
                    if suggestions != textblob:
                        misspelled += 1
            
            # Normalize by word count
            return misspelled / len(words) if words else 0
        except:
            return 0
    
    def _grammar_check(self, text):
        """Simple grammar checking"""
        try:
            sentences = text.split('.')
            if not sentences:
                return 0.5  # Default value
            return 0.5  # Simplified implementation
        except:
            return 0.5
    
    def _sentiment_analysis(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0
    
    def _suspicious_word_ratio(self, text):
        """Calculate ratio of suspicious words in text"""
        words = set(text.lower().split())
        suspicious_count = len(words.intersection(self.suspicious_words))
        if len(words) == 0:
            return 0
        return suspicious_count / len(words)
    
    def _capital_run_length(self, text):
        """Calculate average length of capital letter sequences"""
        capital_runs = re.findall(r'[A-Z]+', text)
        if not capital_runs:
            return 0
        return sum(len(run) for run in capital_runs) / len(capital_runs)
