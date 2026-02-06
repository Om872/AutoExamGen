import nltk

print("Testing NLTK installation...")

# Test tokenization
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    text = "This is a test sentence. NLTK should be able to tokenize this."
    print("\nTokenization test:")
    print(f"Word tokens: {word_tokenize(text)}")
    print(f"Sentences: {sent_tokenize(text)}")
except Exception as e:
    print(f"Tokenization error: {e}")

# Test POS tagging
try:
    from nltk import pos_tag
    tokens = word_tokenize("This is a test")
    print("\nPOS tagging test:")
    print(f"POS tags: {pos_tag(tokens)}")
except Exception as e:
    print(f"POS tagging error: {e}")

# Test stopwords
try:
    from nltk.corpus import stopwords
    print("\nStopwords test:")
    print(f"English stopwords (first 5): {stopwords.words('english')[:5]}")
except Exception as e:
    print(f"Stopwords error: {e}")

print("\nNLTK test complete.")
