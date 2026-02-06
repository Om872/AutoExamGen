import nltk

def download_nltk_data():
    """Download all required NLTK data packages."""
    print("Downloading NLTK data packages...")
    
    # List of NLTK data packages to download
    packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet',
        'stopwords',
        'universal_tagset',
        'tagsets',
        'omw-1.4',  # Open Multilingual WordNet
    ]
    
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=False)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
    
    print("\nNLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()
