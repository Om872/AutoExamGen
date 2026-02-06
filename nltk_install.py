import os
import nltk
import sys

def setup_nltk():
    # Set NLTK data directory
    nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)
    
    print(f"NLTK data directory: {nltk_dir}")
    
    # List of required NLTK packages
    packages = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]
    
    # Download each package
    for package in packages:
        print(f"\nDownloading {package}...")
        try:
            nltk.download(package, download_dir=nltk_dir, quiet=False)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {package}: {str(e)}")
    
    # Verify installation
    print("\n=== Verifying NLTK Installation ===")
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk import pos_tag
        from nltk.corpus import stopwords
        
        test_text = "NLTK is working correctly if you can read this."
        
        # Test tokenization
        print("Testing tokenization...")
        words = word_tokenize(test_text)
        print(f"Word tokens: {words}")
        
        # Test sentence tokenization
        sentences = sent_tokenize(test_text)
        print(f"Sentences: {sentences}")
        
        # Test POS tagging
        print("\nTesting POS tagging...")
        tags = pos_tag(words)
        print(f"POS tags: {tags}")
        
        # Test stopwords
        print("\nTesting stopwords...")
        stop_words = stopwords.words('english')
        print(f"Sample stopwords: {stop_words[:5]}...")
        
        print("\n✅ NLTK is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error verifying NLTK: {str(e)}")
        print("\nPlease try running these commands manually:")
        print("import nltk")
        for package in packages:
            print(f"nltk.download('{package}')")

if __name__ == "__main__":
    setup_nltk()
