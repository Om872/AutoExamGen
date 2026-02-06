import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with required NLTK data."""
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def download_nltk_data(self):
        """Download required NLTK data if not already present."""
        required_data = [
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]
        
        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading NLTK {name}...")
                nltk.download(name)
    
    def clean_text(self, text):
        """
        Clean and preprocess the input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common header/footer patterns (e.g., "Page 1 of 10", "Unit 1")
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Unit\s+\d+(\.\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove standalone numbers (often page numbers or list markers)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters but keep sentence structure
        # Keep periods, question marks, exclamation points, commas, and hyphens
        text = re.sub(r'[^\w\s\.\?\!,\-]', '', text)
        
        # Remove multiple periods/spaces
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_sentences(self, text):
        """
        Tokenize text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text)
        # Filter out very short sentences (less than 5 words)
        filtered_sentences = [s for s in sentences if len(word_tokenize(s)) >= 5]
        return filtered_sentences
    
    def tokenize_words(self, text):
        """
        Tokenize text into words and remove stopwords.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of processed words
        """
        words = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        words = [word for word in words if word not in string.punctuation]
        words = [word for word in words if word not in self.stop_words]
        
        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return words
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Raw input text
            
        Returns:
            dict: Processed text components
        """
        cleaned_text = self.clean_text(text)
        sentences = self.tokenize_sentences(cleaned_text)
        words = self.tokenize_words(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'words': words,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
