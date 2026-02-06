import nltk
import os

def setup_nltk():
    try:
        # Set NLTK data path to a local directory
        nltk_data = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data, exist_ok=True)
        nltk.data.path.append(nltk_data)
        
        # List of NLTK packages to download
        packages = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
            ('wordnet', 'corpora/wordnet'),
            ('omw-1.4', 'corpora/omw-1.4')
        ]
        
        print("\n=== Setting up NLTK data ===")
        
        # Download each package
        for package, path in packages:
            try:
                nltk.data.find(path)
                print(f"✓ {package} is already available")
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, download_dir=nltk_data)
                print(f"✓ Downloaded {package}")
        
        # Test NLTK components
        print("\n=== Testing NLTK Components ===")
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.tag import pos_tag
        from nltk.corpus import stopwords
        
        sent_tokenize("This is a test.")
        word_tokenize("This is a test.")
        pos_tag(["test", "this", "is", "a", "sentence"])
        stopwords.words('english')
        
        print("\n=== NLTK Setup Completed Successfully ===\n")
        return True
        
    except Exception as e:
        print(f"\n⚠ Error during NLTK setup: {str(e)}")
        print("\nPlease try running these commands manually in a Python shell:")
        print("import nltk")
        for package, _ in packages:
            print(f"nltk.download('{package}')")
        return False

if __name__ == "__main__":
    setup_nltk()
