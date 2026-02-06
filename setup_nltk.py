import os
import nltk

def download_nltk_data():
    # Set NLTK data path to a local directory
    nltk_data = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data, exist_ok=True)
    nltk.data.path.append(nltk_data)
    
    print(f"NLTK data will be downloaded to: {nltk_data}")
    
    # List of NLTK packages to download
    packages = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'wordnet',
        'omw-1.4',
        'maxent_ne_chunker',
        'words',
        'punkt'
    ]
    
    print("\n=== Downloading NLTK Data ===")
    
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"⚠ Error downloading {package}: {str(e)}")
    
    print("\n=== NLTK Setup Complete ===")
    print(f"NLTK data location: {nltk_data}")
    print("You can now run your application.")

if __name__ == "__main__":
    download_nltk_data()
