import nltk
import os
import sys
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

def setup_environment():
    """Set up the environment for the phishing email detector"""
    print(f"{Fore.CYAN}Setting up environment for Phishing Email Detector...{Style.RESET_ALL}")
    
    # Create necessary directories
    directories = ["models", "feedback"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Download NLTK resources
    try:
        print(f"{Fore.YELLOW}Downloading NLTK resources...{Style.RESET_ALL}")
        resources = ['punkt', 'stopwords']
        for resource in resources:
            print(f"- Downloading {resource}...")
            nltk.download(resource, quiet=True)
        print(f"{Fore.GREEN}NLTK resources downloaded successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error downloading NLTK resources: {str(e)}{Style.RESET_ALL}")
        print("Please try downloading manually:")
        print("python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        return False
    
    print(f"{Fore.GREEN}Environment setup complete!{Style.RESET_ALL}")
    return True

if __name__ == "__main__":
    setup_environment()
