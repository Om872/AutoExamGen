import re
import random
import os
import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from collections import defaultdict
import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np

# Simple NLTK data setup
def setup_nltk():
    try:
        # Try to import required NLTK components
        sent_tokenize("Test")
        word_tokenize("Test")
        pos_tag(["test"])
        stopwords.words('english')
        return True
    except LookupError:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            return True
        except:
            return False

# Initialize NLTK
if not setup_nltk():
    print("Warning: Could not initialize NLTK. Some features may not work properly.")

# Set up NLTK data path
def setup_nltk():
    try:
        # Set NLTK data path to a local directory
        nltk_data = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data, exist_ok=True)
        nltk.data.path.append(nltk_data)
        
        # Download required NLTK data
        print("\n=== Downloading NLTK Data ===")
        
        # Download punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ punkt tokenizer is already available")
        except LookupError:
            print("Downloading punkt tokenizer...")
            nltk.download('punkt', download_dir=nltk_data)
            print("✓ Downloaded punkt tokenizer")
        
        # Download stopwords
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ Stopwords are already available")
        except LookupError:
            print("Downloading stopwords...")
            nltk.download('stopwords', download_dir=nltk_data)
            print("✓ Downloaded stopwords")
        
        # Download averaged_perceptron_tagger
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
            print("✓ POS tagger is already available")
        except LookupError:
            print("Downloading POS tagger...")
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data)
            print("✓ Downloaded POS tagger")
        
        # Download wordnet
        try:
            nltk.data.find('corpora/wordnet')
            print("✓ WordNet is already available")
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet', download_dir=nltk_data)
            print("✓ Downloaded WordNet")
        
        # Download omw-1.4
        try:
            nltk.data.find('corpora/omw-1.4')
            print("✓ OMW-1.4 is already available")
        except LookupError:
            print("Downloading OMW-1.4...")
            nltk.download('omw-1.4', download_dir=nltk_data)
            print("✓ Downloaded OMW-1.4")
        
        # Test NLTK components
        print("\n=== Testing NLTK Components ===")
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
        print("nltk.download('punkt')")
        print("nltk.download('stopwords')")
        print("nltk.download('averaged_perceptron_tagger')")
        print("nltk.download('wordnet')")
        print("nltk.download('omw-1.4')\n")
        return False

# Initialize NLTK
if not setup_nltk():
    print("Failed to initialize NLTK. Some features may not work properly.")
    print("Trying to continue with limited functionality...\n")
    try:
        print(f"✓ {package} is already downloaded")
    except LookupError:
            print(f"Downloading {package}...")
            try:
                nltk.download(package, download_dir=nltk_data, quiet=False)
                # Verify download
                try:
                    nltk.data.find(path)
                    print(f"✓ Successfully downloaded {package}")
                except LookupError:
                    print(f"⚠ Warning: {package} download verification failed")
            except Exception as e:
                print(f"⚠ Error downloading {package}: {str(e)}")
                if package == 'averaged_perceptron_tagger':
                    print("⚠ This is a critical package. The application may not work properly.")
    
    print("\n=== NLTK Data Setup Complete ===\n")

# Initialize NLTK data
download_nltk_data()

# Initialize NLTK components
try:
    # Initialize tokenizers
    sent_tokenize("Initializing...")
    word_tokenize("Initializing...")
    
    # Initialize POS tagger
    from nltk.tag import pos_tag
    pos_tag(["test"])
    
    # Initialize stopwords
    stopwords.words('english')
    
    print("✓ NLTK components initialized successfully")
except Exception as e:
    print(f"⚠ Error initializing NLTK components: {str(e)}")

class QuestionGenerator:
    def __init__(self, model_name="deepset/roberta-base-squad2", use_transformers=True):
        """
        Initialize the question generator with improved context understanding.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            use_transformers (bool): Whether to use transformer models for better quality
        """
        print("Initializing question generator with enhanced context understanding...")
        self.use_transformers = use_transformers
        self.stop_words = set(stopwords.words('english'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use_transformers:
            try:
                print("Loading question generation model...")
                self.qg_model = pipeline("text2text-generation", 
                                      model="valhalla/t5-base-qa-qg-hl",
                                      device=0 if self.device == 'cuda' else -1)
                print("Question generation model loaded successfully!")
            except Exception as e:
                print(f"Error loading transformer model: {str(e)}")
                print("Falling back to rule-based generation.")
                self.use_transformers = False
        
        if not self.use_transformers:
            print("Using rule-based question generation.")
            self._init_rule_based_system()
            
        print("Question generator initialized successfully!")
    
    def _init_rule_based_system(self):
        """Initialize the rule-based question generation system."""
        self.wh_words = ['what', 'when', 'where', 'who', 'whom', 'whose', 'which', 'why', 'how']
        self.aux_verbs = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must']
        self.important_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VBG', 'VBN', 'JJ', 'JJR', 'JJS'}
        
    def _extract_key_phrases(self, text):
        """Extract key phrases from text based on POS tagging."""
        words = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        key_phrases = []
        current_phrase = []
        
        for word, tag in pos_tags:
            if tag in self.important_pos_tags:
                current_phrase.append(word.lower())
            elif current_phrase:
                if len(current_phrase) > 1:  # Only consider phrases with at least 2 words
                    key_phrases.append(' '.join(current_phrase))
                current_phrase = []
        
        return list(set(key_phrases))  # Remove duplicates

    def generate_question_from_sentence(self, sentence):
        """Generate a question from a given sentence using rule-based approach."""
        words = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        
        # Find the main verb and subject
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB'):  # Verb
                # Find the subject before the verb
                for j in range(i-1, -1, -1):
                    if pos_tags[j][1].startswith('NN'):  # Noun
                        subject = ' '.join([w for w, _ in pos_tags[j:i]])
                        # Create a wh-question
                        question = f"What {pos_tags[i][0]} {subject}?"
                        return question
        
        # Fallback: create a what question about the main noun phrase
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('NN'):  # Noun
                return f"What is {word}?"
        
        # Final fallback
        return f"What is the main idea of: {sentence[:50]}...?"

    def _analyze_text_structure(self, text):
        """Analyze text structure to identify important concepts and relationships."""
        sentences = sent_tokenize(text)
        key_phrases = self._extract_key_phrases(text)
        
        # Find most important terms using frequency distribution
        words = [word.lower() for word in word_tokenize(text) 
                if word.isalnum() and word.lower() not in self.stop_words]
        freq_dist = FreqDist(words)
        
        return {
            'sentences': sentences,
            'key_phrases': key_phrases,
            'top_terms': [word for word, _ in freq_dist.most_common(10)],
            'concept_map': self._build_concept_map(sentences, key_phrases)
        }
        
    def _build_concept_map(self, sentences, key_phrases):
        """Build a simple concept map showing relationships between key phrases."""
        concept_map = defaultdict(list)
        
        for phrase in key_phrases:
            for sentence in sentences:
                if phrase in sentence.lower():
                    # Find other key phrases in the same sentence
                    related = [p for p in key_phrases if p != phrase and p in sentence.lower()]
                    concept_map[phrase].extend(related)
        
        # Remove duplicates
        return {k: list(set(v)) for k, v in concept_map.items()}
        try:
            print("Attempting to load T5 model... This may take a few minutes on first run.")
            
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            # Use smaller, faster model for web deployment
            model_name = "t5-base"
            
            print(f"Loading {model_name} model...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Use CPU for more reliable deployment (avoid CUDA issues)
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.use_transformers = True
            print(f"T5 model loaded successfully on {self.device}")
            
        except ImportError as e:
            print(f"Transformers library not installed: {e}")
            print("Install with: pip install transformers torch")
            self.use_transformers = False
        except Exception as e:
            print(f"Failed to load T5 model: {e}")
            print("Falling back to rule-based generation.")
            self.use_transformers = False
    
    def generate_questions(self, text, num_questions=5, context_window=3):
        """
        Generate meaningful questions from the given text with better context understanding.
        
        Args:
            text (str): Input text to generate questions from
            num_questions (int): Number of questions to generate
            context_window (int): Number of sentences to consider as context
            
        Returns:
            list: List of generated questions with their context
        """
        if not text.strip():
            return []
        
        # Analyze the text structure first
        analysis = self._analyze_text_structure(text)
        sentences = analysis['sentences']
        
        if not sentences:
            return []
        
        questions = []
        
        # Generate questions using different strategies
        if self.use_transformers and hasattr(self, 'qg_model'):
            # Use transformer-based generation for better quality
            for i in range(0, len(sentences), context_window):
                context = ' '.join(sentences[i:i+context_window])
                try:
                    # Generate questions for this context window
                    generated = self.qg_model(context, max_length=128, num_return_sequences=1)
                    if generated and len(generated) > 0:
                        question = generated[0]['generated_text'].strip()
                        if question and question[-1] != '?':
                            question += '?'
                        questions.append({
                            'question': question,
                            'context': context,
                            'type': 'comprehension'
                        })
                        if len(questions) >= num_questions:
                            break
                except Exception as e:
                    print(f"Error in transformer-based generation: {str(e)}")
                    continue
        
        # Fallback to rule-based generation if needed
        if len(questions) < num_questions:
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 5:  # Skip very short sentences
                    continue
                    
                # Generate question using rule-based approach
                question = self._generate_question_from_sentence(sentence)
                
                # Get context (previous and next sentences)
                start = max(0, i-1)
                end = min(len(sentences), i+2)
                context = ' '.join(sentences[start:end])
                
                questions.append({
                    'question': question,
                    'context': context,
                    'type': 'factual'
                })
                
                if len(questions) >= num_questions:
                    break
        
        # Ensure we have enough questions
        if len(questions) < num_questions:
            # Generate some conceptual questions based on key phrases
            for phrase in analysis['key_phrases'][:num_questions - len(questions)]:
                questions.append({
                    'question': f"Explain the concept of {phrase} in detail.",
                    'context': f"The concept of {phrase} is important in this context.",
                    'type': 'conceptual'
                })
        
        return questions[:num_questions]
    
    def _generate_with_transformers(self, sentence, max_length):
        """Generate question using T5 model."""
        if not self.use_transformers or self.model is None:
            return self._generate_with_rules(sentence)
        
        try:
            # Prepare input for T5 model
            input_text = f"generate question: {sentence[:300]}"  # Limit input length
            
            # Tokenize input with error handling
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=256,  # Reduced for faster processing
                truncation=True,
                padding=True
            )
            
            if self.device:
                inputs = inputs.to(self.device)
            
            # Generate question with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(max_length, 64),  # Increased output length
                    num_beams=4,  # Increased beams for better quality
                    early_stopping=True,
                    do_sample=False,  # Deterministic for consistency
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean question
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_question = self.clean_question(question)
            
            # Validate the generated question
            if len(cleaned_question) < 10 or not cleaned_question.endswith('?'):
                print("Generated question quality low, using rule-based fallback")
                return self._generate_with_rules(sentence)
            
            return cleaned_question
            
        except Exception as e:
            print(f"Transformer generation failed: {e}")
            print("Falling back to rule-based generation")
            return self._generate_with_rules(sentence)
    
    def _generate_with_rules(self, sentence):
        """Generate question using rule-based approach."""
        sentence = sentence.strip()
        words = sentence.split()
        
        # Enhanced question templates based on sentence patterns
        question_templates = [
            # What questions - most common
            (lambda s: any(word in s.lower() for word in ['is', 'are', 'means', 'refers', 'definition', 'concept']), 
             lambda s: f"What {self._extract_predicate(s)}?"),
            
            # Define/Explain questions
            (lambda s: any(word in s.lower() for word in ['definition', 'meaning', 'concept', 'term']),
             lambda s: f"Define {self._extract_main_subject(s)}."),
            
            # How questions
            (lambda s: any(word in s.lower() for word in ['process', 'method', 'way', 'procedure', 'algorithm']),
             lambda s: f"How {self._extract_predicate(s)}?"),
            
            # Why questions
            (lambda s: any(word in s.lower() for word in ['because', 'reason', 'cause', 'purpose', 'important']),
             lambda s: f"Why {self._extract_predicate(s)}?"),
            
            # When questions
            (lambda s: any(word in s.lower() for word in ['year', 'century', 'time', 'date', 'period', 'era']),
             lambda s: f"When {self._extract_predicate(s)}?"),
            
            # Where questions
            (lambda s: any(word in s.lower() for word in ['place', 'location', 'country', 'city', 'region']),
             lambda s: f"Where {self._extract_predicate(s)}?"),
            
            # Who questions
            (lambda s: any(word in s.lower() for word in ['person', 'people', 'scientist', 'author', 'researcher']),
             lambda s: f"Who {self._extract_predicate(s)}?"),
            
            # How questions
            (lambda s: any(word in s.lower() for word in ['method', 'process', 'way', 'how']),
             lambda s: f"How {self._extract_predicate(s)}?"),
            
            # Why questions
            (lambda s: any(word in s.lower() for word in ['reason', 'because', 'cause', 'why']),
             lambda s: f"Why {self._extract_predicate(s)}?"),
            
            # Default question
            (lambda s: True,
             lambda s: f"What can you tell me about {self._extract_main_subject(s)}?")
        ]
        
        # Apply first matching template
        for condition, template in question_templates:
            if condition(sentence):
                try:
                    question = template(sentence)
                    return self.clean_question(question)
                except:
                    continue
        
        # Fallback
        return f"What is the main point about {words[0] if words else 'this topic'}?"
    
    def _extract_main_subject(self, sentence):
        """Extract the main subject from a sentence."""
        words = sentence.split()
        # Look for capitalized words (likely proper nouns)
        subjects = [word.strip('.,!?;:') for word in words if word[0].isupper() and len(word) > 2]
        if subjects:
            return subjects[0]
        # Fallback to first few words
        return ' '.join(words[:3]) if len(words) >= 3 else sentence
    
    def _extract_predicate(self, sentence):
        """Extract predicate for question formation."""
        sentence = sentence.lower()
        # Remove common sentence starters
        sentence = re.sub(r'^(the|this|that|these|those|a|an)\s+', '', sentence)
        
        # Find verb patterns
        if ' is ' in sentence:
            parts = sentence.split(' is ', 1)
            if len(parts) > 1:
                return f"is {parts[1].strip('.,!?;:')}"
        
        if ' are ' in sentence:
            parts = sentence.split(' are ', 1)
            if len(parts) > 1:
                return f"are {parts[1].strip('.,!?;:')}"
        
        # Default fallback
        words = sentence.split()
        if len(words) > 3:
            return ' '.join(words[1:]).strip('.,!?;:')
        return sentence.strip('.,!?;:')
    
    def clean_question(self, question):
        """
        Clean and format the generated question.
        
        Args:
            question (str): Raw generated question
            
        Returns:
            str: Cleaned question
        """
        # Remove extra spaces
        question = re.sub(r'\s+', ' ', question.strip())
        
        # Ensure question ends with question mark
        if not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def generate_multiple_questions(self, sentences, max_questions=5):
        """
        Generate multiple questions from a list of sentences.
        
        Args:
            sentences (list): List of sentences to generate questions from
            max_questions (int): Maximum number of questions to generate
            
        Returns:
            list: List of generated questions with their source sentences
        """
        questions = []
        
        for i, (score, sentence) in enumerate(sentences[:max_questions]):
            try:
                question = self.generate_question_from_sentence(sentence)
                
                # Filter out low-quality questions
                if self.is_valid_question(question):
                    questions.append({
                        'question': question,
                        'context': sentence,
                        'score': score,
                        'question_id': i + 1
                    })
            except Exception as e:
                print(f"Error generating question from sentence: {sentence[:50]}... Error: {e}")
                continue
        
        return questions
    
    def is_valid_question(self, question):
        """
        Check if a generated question is valid.
        
        Args:
            question (str): Generated question
            
        Returns:
            bool: True if question is valid
        """
        # Basic validation criteria
        if len(question) < 10:  # Too short
            return False
        
        if len(question) > 200:  # Too long
            return False
        
        # Must contain question words or end with question mark
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']
        question_lower = question.lower()
        
        has_question_word = any(word in question_lower for word in question_words)
        ends_with_question_mark = question.endswith('?')
        
        return has_question_word or ends_with_question_mark
