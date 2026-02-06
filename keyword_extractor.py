import nltk
from rake_nltk import Rake
from collections import Counter
import re

class KeywordExtractor:
    def __init__(self):
        """Initialize the keyword extractor."""
        self.rake = Rake()
    
    def extract_keywords_rake(self, text, max_keywords=10):
        """
        Extract keywords using RAKE algorithm.
        
        Args:
            text (str): Input text
            max_keywords (int): Maximum number of keywords to extract
            
        Returns:
            list: List of keywords with scores
        """
        self.rake.extract_keywords_from_text(text)
        keywords_with_scores = self.rake.get_ranked_phrases_with_scores()
        
        # Filter keywords
        clean_keywords = []
        seen_keywords = set()
        
        for score, keyword in keywords_with_scores:
            # Remove keywords with digits or special chars
            if re.search(r'\d', keyword) or len(keyword) < 4:
                continue
                
            # Remove very short single words that are lowercase (likely noise)
            if ' ' not in keyword and keyword[0].islower() and len(keyword) < 5:
                continue
            
            # Remove duplicates
            if keyword.lower() in seen_keywords:
                continue
                
            clean_keywords.append((score, keyword))
            seen_keywords.add(keyword.lower())
            
            if len(clean_keywords) >= max_keywords:
                break
        
        return clean_keywords
    
    def extract_named_entities(self, text):
        """
        Extract named entities (simple approach using capitalization patterns).
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of potential named entities
        """
        # Simple named entity extraction based on capitalization
        words = text.split()
        entities = []
        
        for word in words:
            # Look for capitalized words that aren't at sentence start
            if word[0].isupper() and len(word) > 2:
                # Remove punctuation
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and clean_word not in ['The', 'This', 'That', 'These', 'Those']:
                    entities.append(clean_word)
        
        # Count occurrences and return most frequent
        entity_counts = Counter(entities)
        return entity_counts.most_common(10)
    
    def identify_important_sentences(self, sentences, keywords, top_n=5):
        """
        Identify important sentences based on keyword density.
        
        Args:
            sentences (list): List of sentences
            keywords (list): List of important keywords
            top_n (int): Number of top sentences to return
            
        Returns:
            list: List of important sentences with scores
        """
        keyword_phrases = [kw[1] for kw in keywords]  # Extract phrases from (score, phrase) tuples
        sentence_scores = []
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            best_keyword = ""
            
            # Score based on keyword presence
            for keyword in keyword_phrases:
                if keyword.lower() in sentence_lower:
                    score += 1
                    if not best_keyword or len(keyword) > len(best_keyword):
                        best_keyword = keyword
            
            # Bonus for sentence length (not too short, not too long)
            word_count = len(sentence.split())
            if 8 <= word_count <= 25:
                score += 0.5
            
            # Bonus for sentences with numbers or specific terms
            if re.search(r'\d+', sentence):
                score += 0.3
            
            if score > 0:
                sentence_scores.append((score, sentence, best_keyword))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        return sentence_scores[:top_n]
    
    def extract_key_concepts(self, text, sentences, top_n_sentences=5):
        """
        Complete keyword and concept extraction pipeline.
        
        Args:
            text (str): Input text
            sentences (list): List of sentences
            
        Returns:
            dict: Extracted keywords, entities, and important sentences
        """
        keywords = self.extract_keywords_rake(text)
        entities = self.extract_named_entities(text)
        important_sentences = self.identify_important_sentences(sentences, keywords, top_n=top_n_sentences)
        
        return {
            'keywords': keywords,
            'named_entities': entities,
            'important_sentences': important_sentences
        }
