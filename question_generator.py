import re
import nltk
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK data setup
def setup_nltk():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        logger.info("NLTK data is already set up.")
        return True
    except LookupError:
        logger.info("Downloading required NLTK data...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            logger.info("NLTK data downloaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {str(e)}")
            return False

# Initialize NLTK
if not setup_nltk():
    logger.warning("NLTK data not available. Some features may not work properly.")

class QuestionGenerator:
    def __init__(self, model_name="valhalla/t5-small-qa-qg-hl", use_transformers=False):
        """Initialize the question generator with enhanced capabilities."""
        self.use_transformers = use_transformers
        self.stop_words = set(stopwords.words('english'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize rule-based system
        self._init_rule_based_system()
        
        # Initialize transformer model if requested
        if use_transformers:
            try:
                logger.info("Loading transformer model...")
                self.qg_model = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=0 if self.device == 'cuda' else -1
                )
                logger.info("Transformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading transformer model: {str(e)}")
                self.use_transformers = False
                logger.info("Falling back to rule-based generation.")
    
    def _init_rule_based_system(self):
        """Initialize the rule-based question generation system."""
        self.wh_words = ['what', 'when', 'where', 'who', 'whom', 'whose', 'which', 'why', 'how']
        self.aux_verbs = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had',
                         'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must']
        self.common_nouns = {'time', 'year', 'people', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child',
                           'world', 'school', 'state', 'family', 'student', 'group', 'country', 'problem'}
    
    def _is_good_sentence(self, sentence):
        """Check if a sentence is suitable for question generation."""
        try:
            if not sentence or not isinstance(sentence, str):
                return False
                
            # Basic length checks
            words = word_tokenize(sentence)
            if len(words) < 4:  # At least 4 words
                return False
                
            # Check for question mark
            if '?' in sentence:
                return False
                
            # Check for proper sentence ending
            if not sentence.strip().endswith(('.', '!', ';', ':')):
                return False
                
            # Check for at least one noun and one verb
            pos_tags = pos_tag(words)
            has_noun = any(tag.startswith('NN') for word, tag in pos_tags)
            has_verb = any(tag.startswith('VB') for word, tag in pos_tags)
            
            return has_noun and has_verb
            
        except Exception as e:
            logger.error(f"Error in _is_good_sentence: {str(e)}")
            return False
    
    def _generate_question_what_is(self, words, pos_tags):
        """Generate 'What is...?' questions."""
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('NN'):
                return f"What is {word}?"
        return ""
    
    def _generate_question_verb_subject(self, words, pos_tags):
        """Generate questions by inverting subject and verb."""
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB') and i > 0:
                subject = ' '.join(words[:i])
                verb = word
                rest = ' '.join(words[i+1:])
                return f"{verb.capitalize()} {subject} {rest}?"
        return ""
    
    def _generate_question_wh_word(self, words, pos_tags):
        """Generate questions using WH-words."""
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB') and i > 0:
                wh_word = "What"
                if i > 0 and pos_tags[i-1][1].startswith('NNP'):
                    wh_word = "Who"
                return f"{wh_word} {word} {' '.join(words[:i])}?"
        return ""
    
    def _generate_question_from_statement(self, sentence):
        """Generate a question from a statement using multiple strategies."""
        try:
            if not sentence or not isinstance(sentence, str):
                return ""
                
            # Clean the sentence
            sentence = sentence.strip()
            if sentence.endswith('.'):
                sentence = sentence[:-1].strip()
                
            words = word_tokenize(sentence)
            if len(words) < 4:  # Too short for a good question
                return ""
                
            pos_tags = pos_tag(words)
            
            # Try different question generation strategies
            strategies = [
                self._generate_question_what_is,
                self._generate_question_verb_subject,
                self._generate_question_wh_word
            ]
            
            for strategy in strategies:
                question = strategy(words, pos_tags)
                if question:
                    return question
            
            # Fallback: ask about the whole sentence
            return f"Can you explain: {sentence[:100]}...?"
            
        except Exception as e:
            logger.error(f"Error in _generate_question_from_statement: {str(e)}")
            return ""
    
    def generate_question_from_sentence(self, sentence):
        """Generate a question from a given sentence."""
        if not self._is_good_sentence(sentence):
            return ""
            
        try:
            # Use transformer model if available
            if self.use_transformers and hasattr(self, 'qg_model'):
                try:
                    # Prepare input for e2e model
                    input_text = f"generate questions: {sentence}"
                    outputs = self.qg_model(input_text)
                    if outputs and len(outputs) > 0:
                        generated_text = outputs[0]['generated_text']
                        # The model might generate multiple questions separated by <sep>
                        questions = generated_text.split('<sep>')
                        if questions:
                            return questions[0].strip()
                except Exception as e:
                    logger.error(f"Transformer generation failed: {e}")
                    # Fallback to rule-based

            # First try rule-based generation
            question = self._generate_question_from_statement(sentence)
            if question:
                return question
                
            # Fallback to simple question generation
            words = word_tokenize(sentence)
            if len(words) < 4:
                return ""
                
            # Try to make a simple question
            return f"What is the main point about: {sentence[:100]}...?"
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return ""
    
    def _score_sentence(self, sentence):
        """Score a sentence based on its quality for question generation."""
        try:
            if not self._is_good_sentence(sentence):
                return 0
                
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            
            # Start with base score
            score = 1.0
            
            # Check for content words
            has_noun = any(tag.startswith('NN') for _, tag in pos_tags)
            has_verb = any(tag.startswith('VB') for _, tag in pos_tags)
            has_adj = any(tag.startswith('JJ') for _, tag in pos_tags)
            
            # Increase score based on content
            if has_noun and has_verb:
                score += 2.0
            elif has_noun or has_verb:
                score += 1.0
                
            if has_adj:
                score += 0.5
            
            # Adjust for sentence length
            word_count = len(words)
            if 8 <= word_count <= 25:  # Ideal length
                score += 1.0
            
            # Bonus for proper nouns or numbers
            if any(tag in {'NNP', 'NNPS', 'CD'} for _, tag in pos_tags):
                score += 1.0
                
            return max(0.5, score)  # Ensure minimum score
            
        except Exception as e:
            logger.error(f"Error in _score_sentence: {str(e)}")
            return 0.5
    
    def generate_questions(self, text, num_questions=5):
        """Generate questions from the given text."""
        if not text or not text.strip():
            logger.warning("Empty text provided for question generation")
            return []
            
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            return self.generate_multiple_questions(sentences, num_questions)
            
        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}")
            return []
    
    def generate_multiple_questions(self, inputs, max_questions=5):
        """
        Generate multiple questions from a list of inputs (context/answer pairs).
        
        Args:
            inputs: List of dicts {'context': str, 'answer': str} or list of strings
            max_questions: Maximum number of questions to generate
            
        Returns:
            List of generated questions with metadata
        """
        if not inputs or max_questions <= 0:
            logger.warning("No inputs provided or invalid max_questions")
            return []
            
        questions = []
        used_contexts = set()
        
        logger.info(f"Generating up to {max_questions} questions from {len(inputs)} inputs")
        
        for item in inputs:
            try:
                if len(questions) >= max_questions:
                    break
                
                # Handle different input types
                if isinstance(item, dict):
                    context = item.get('context', '')
                    answer = item.get('answer')
                else:
                    context = str(item)
                    answer = None
                    
                if not context or not context.strip():
                    continue
                    
                context = context.strip()
                
                # Skip if we've already used this context
                if context in used_contexts:
                    continue
                    
                question_text = ""
                
                # Use transformer model if available
                if self.use_transformers and hasattr(self, 'qg_model'):
                    try:
                        if answer:
                            input_text = f"answer: {answer} context: {context}"
                        else:
                            input_text = f"generate questions: {context}"
                            
                        outputs = self.qg_model(input_text)
                        if outputs and len(outputs) > 0:
                            question_text = outputs[0]['generated_text']
                    except Exception as e:
                        logger.error(f"Transformer generation failed: {e}")
                
                # Fallback to rule-based if transformer failed or not available
                if not question_text:
                    question_text = self.generate_question_from_sentence(context)
                
                if question_text and question_text not in [q.get('question', '') for q in questions]:
                    q_data = {
                        'question': question_text,
                        'context': context,
                        'score': 1.0,
                        'type': 'short_answer'
                    }
                    # If we have a known answer, use it for options later
                    if answer:
                        q_data['correct_answer'] = answer
                        
                    questions.append(q_data)
                    used_contexts.add(context)
                    
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")
                continue
        
        # If we still don't have enough questions, create simple ones
        if len(questions) < max_questions:
            logger.info(f"Creating simple questions to reach {max_questions} total")
            for i in range(len(questions), max_questions):
                # Try to find an unused context or reuse one
                fallback_context = "General knowledge about the topic"
                if inputs:
                    # Pick a random input to generate a question from
                    import random
                    item = random.choice(inputs)
                    if isinstance(item, dict):
                        fallback_context = item.get('context', fallback_context)
                    else:
                        fallback_context = str(item)
                
                # Create a more specific fallback question
                words = fallback_context.split()
                topic_snippet = " ".join(words[:5]) + "..." if len(words) > 5 else fallback_context
                
                questions.append({
                    'question': f"Explain the significance of: {topic_snippet}",
                    'context': fallback_context,
                    'score': 0.5,
                    'type': 'short_answer'
                })
        
        logger.info(f"Successfully generated {len(questions)} questions")
        return questions[:max_questions]

# Example usage
if __name__ == "__main__":
    # Test the question generator
    qg = QuestionGenerator(use_transformers=False)
    
    test_text = """
    Machine learning is a branch of artificial intelligence that focuses on building systems 
    that learn from data. These systems can improve their performance over time without being 
    explicitly programmed. There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning.
    """
    
    print("\nGenerating questions...")
    questions = qg.generate_questions(test_text, 3)
    
    print("\nGenerated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q.get('question', 'No question generated')}")
        print(f"   Context: {q.get('context', 'No context')[:100]}...")
        print()
