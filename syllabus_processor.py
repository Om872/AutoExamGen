import re
import nltk
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from question_generator import QuestionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyllabusProcessor:
    def __init__(self):
        """Initialize the SyllabusProcessor with necessary NLTK components."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            # Initialize NLTK components
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Import and initialize the PerceptronTagger
            from nltk.tag import PerceptronTagger
            self.tagger = PerceptronTagger()
            
            # Initialize question generator
            self.question_generator = QuestionGenerator()
            
            logger.info("SyllabusProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SyllabusProcessor: {str(e)}")
            raise

    def parse_syllabus(self, syllabus_text: str) -> Dict[str, List[str]]:
        """
        Parse a syllabus text into topics and subtopics.
        
        Args:
            syllabus_text: Raw syllabus text with units and topics
            
        Returns:
            Dictionary mapping unit names to lists of topics
        """
        units = {}
        current_unit = "General Topics"
        units[current_unit] = []
        
        # Split into lines and process each line
        for line in syllabus_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for unit headers (e.g., "Unit 1.0 Introduction" or "Unit 1: Introduction")
            unit_match = re.match(r'(?:Unit[\s\-]*\d+(?:\.\d+)?\s*[:\-]?\s*)(.+)', line, re.IGNORECASE)
            if unit_match:
                current_unit = unit_match.group(0).strip()
                units[current_unit] = []
                
                # Check if there are topics on the same line (e.g., "Unit 1 ... 1.1 Topic")
                remaining_text = unit_match.group(1)
                # Find all topic patterns like "1.1 Topic Name"
                inline_topics = re.findall(r'(\d+(?:\.\d+)+[\s\-]+[^0-9]+)', remaining_text)
                for t in inline_topics:
                    # Clean up the topic text
                    t = re.sub(r'\s*\d+\.\d+.*$', '', t).strip() # Remove next topic number if caught
                    if t:
                        units[current_unit].append(t.strip())
                continue
                
            # Check for topic lines (e.g., "1.1 Topic Name" or "- Topic Name")
            # Handle multiple topics on one line
            topics_on_line = re.findall(r'(\d+(?:\.\d+)+[\s\-]+[^0-9]+)', line)
            if topics_on_line:
                for t in topics_on_line:
                    t = t.strip()
                    # Clean up trailing dots or next topic numbers
                    t = re.sub(r'\s*\d+\.\d+.*$', '', t).strip()
                    if t and len(t) > 3: # Avoid just numbers
                        units[current_unit].append(t)
            else:
                # Check for bullet points
                topic_match = re.match(r'(?:[-•*]\s*)(.+)', line)
                if topic_match:
                    topic = topic_match.group(1).strip()
                    if topic and topic.lower() not in ['introduction', 'overview']:
                        units[current_unit].append(topic)
        
        return units
    
    def extract_key_terms(self, topic: str) -> List[str]:
        """
        Extract key terms from a topic for question generation.
        
        Args:
            topic: The topic text
            
        Returns:
            List of important terms from the topic
        """
        try:
            # Use the instance tagger
            words = word_tokenize(topic.lower())
            pos_tags = self.tagger.tag(words)
            
            # Extract nouns and proper nouns
            key_terms = [
                word for word, tag in pos_tags 
                if tag.startswith('NN') and word not in self.stop_words
            ]
            
            return list(set(key_terms))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}")
            return []

    def generate_topic_based_questions(self, syllabus_text: str, content_text: str, 
                                     questions_per_topic: int = 3) -> Dict[str, List[Dict]]:
        """
        Generate questions based on syllabus topics.
        
        Args:
            syllabus_text: The syllabus text with units and topics
            content_text: The content text to generate questions from
            questions_per_topic: Number of questions to generate per topic
            
        Returns:
            Dictionary mapping topics to lists of questions
        """
        # Parse the syllabus
        units = self.parse_syllabus(syllabus_text)
        
        # Process content into sentences
        sentences = sent_tokenize(content_text)
        
        topic_questions = {}
        
        for unit, topics in units.items():
            for topic in topics:
                # Extract key terms from the topic
                key_terms = self.extract_key_terms(topic)
                
                # Find relevant sentences containing these terms
                relevant_sentences = []
                for sentence in sentences:
                    if any(term in sentence.lower() for term in key_terms):
                        relevant_sentences.append(sentence)
                
                # If no relevant sentences found, use general content
                if not relevant_sentences:
                    relevant_sentences = sentences
                
                # Generate questions from relevant sentences
                questions = self.question_generator.generate_multiple_questions(
                    relevant_sentences, 
                    max_questions=min(questions_per_topic, len(relevant_sentences))
                )
                
                if questions:
                    topic_questions[f"{unit} - {topic}"] = questions
        
        return topic_questions