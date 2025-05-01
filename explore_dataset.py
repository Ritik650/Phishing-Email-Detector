# explore_dataset.py
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import argparse

def clean_text(text):
    """Clean text for analysis"""
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_dataset(visualize=True):
    """Analyze the phishing email dataset"""
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("zefang-liu/phishing-email-dataset")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Print basic info
    print("\n===== DATASET SUMMARY =====")
    print(f"Total emails: {len(df)}")
    
    # Distribution of labels
    label_counts = df['Email Type'].value_counts()
    print("\n===== LABEL DISTRIBUTION =====")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(df)*100:.2f}%)")
    
    # Basic statistics
    print("\n===== EMAIL STATISTICS =====")
    df['Email Length'] = df['Email Text'].apply(lambda x: len(str(x)))
    df['Word Count'] = df['Email Text'].apply(lambda x: len(str(x).split()))
    
    print(f"Average email length: {df['Email Length'].mean():.2f} characters")
    print(f"Average word count: {df['Word Count'].mean():.2f} words")
    
    print(f"Average phishing email length: {df[df['Email Type'] == 'Phishing Email']['Email Length'].mean():.2f} characters")
    print(f"Average safe email length: {df[df['Email Type'] == 'Safe Email']['Email Length'].mean():.2f} characters")
    
    # URL analysis
    print("\n===== URL ANALYSIS =====")
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    df['URL Count'] = df['Email Text'].apply(lambda x: len(url_pattern.findall(str(x))))
    
    print(f"Emails containing URLs: {len(df[df['URL Count'] > 0])} ({len(df[df['URL Count'] > 0])/len(df)*100:.2f}%)")
    print(f"Phishing emails with URLs: {len(df[(df['Email Type'] == 'Phishing Email') & (df['URL Count'] > 0)])} "
          f"({len(df[(df['Email Type'] == 'Phishing Email') & (df['URL Count'] > 0)])/len(df[df['Email Type'] == 'Phishing Email'])*100:.2f}%)")
    print(f"Safe emails with URLs: {len(df[(df['Email Type'] == 'Safe Email') & (df['URL Count'] > 0)])} "
          f"({len(df[(df['Email Type'] == 'Safe Email') & (df['URL Count'] > 0)])/len(df[df['Email Type'] == 'Safe Email'])*100:.2f}%)")
    
    # Common words analysis
    if not visualize:
        return df
    
    print("\n===== VISUALIZATION =====")
    print("Generating visualizations...")
    
    # Label distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Email Type')
    plt.title('Distribution of Email Types')
    plt.savefig('email_distribution.png')
    print("Saved email distribution chart to 'email_distribution.png'")
    
    # Email length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Email Length', hue='Email Type', kde=True, bins=50)
    plt.title('Distribution of Email Lengths')
    plt.xlim(0, df['Email Length'].quantile(0.95))  # Limit to 95th percentile for better visualization
    plt.savefig('email_length_distribution.png')
    print("Saved email length distribution chart to 'email_length_distribution.png'")
    
    # URL count comparison
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='URL Count', hue='Email Type', bins=10, discrete=True)
    plt.title('URL Count by Email Type')
    plt.xlim(0, df['URL Count'].quantile(0.95))  # Limit to 95th percentile
    plt.savefig('url_count_distribution.png')
    print("Saved URL count distribution chart to 'url_count_distribution.png'")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze phishing email dataset')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    args = parser.parse_args()
    
    analyze_dataset(visualize=not args.no_viz)
