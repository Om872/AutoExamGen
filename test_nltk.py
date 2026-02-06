import nltk
import os

# Set NLTK data path
def setup_nltk():
    nltk_data = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data, exist_ok=True)
    nltk.data.path.append(nltk_data)
    return nltk_data

print("Testing NLTK installation...")
try:
    # Test basic NLTK functionality
    print("Testing tokenizers...")
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    text = "This is a test sentence. NLTK is working!"
    print(f"Word tokens: {word_tokenize(text)}")
    print(f"Sentences: {sent_tokenize(text)}")
    
    print("\nTesting POS tagging...")
    from nltk import pos_tag
    print(f"POS tags: {pos_tag(word_tokenize(text))}")
    
    print("\nTesting stopwords...")
    from nltk.corpus import stopwords
    print(f"English stopwords: {list(stopwords.words('english'))[:5]}...")
    
    print("\n✅ NLTK is working correctly!")
    
except LookupError as e:
    print(f"\n❌ NLTK data not found: {e}")
    nltk_data = setup_nltk()
    print(f"\nPlease run these commands in a Python shell to download NLTK data:")
    print(f"import nltk")
    print(f"nltk.download('punkt', download_dir=r'{nltk_data}')")
    print(f"nltk.download('stopwords', download_dir=r'{nltk_data}')")
    print(f"nltk.download('averaged_perceptron_tagger', download_dir=r'{nltk_data}')")
    print(f"nltk.download('wordnet', download_dir=r'{nltk_data}')")
    print(f"nltk.download('omw-1.4', download_dir=r'{nltk_data}')")
    
except Exception as e:
    print(f"\n❌ Error testing NLTK: {e}")
