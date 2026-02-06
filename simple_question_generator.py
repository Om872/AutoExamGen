import re
import random

class SimpleQuestionGenerator:
    def __init__(self):
        self.question_words = [
            'What', 'Why', 'How', 'When', 'Where', 'Who', 'Which', 'Describe', 'Explain'
        ]
        self.auxiliary_verbs = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had']
    
    def generate_question(self, sentence):
        """Generate a simple question from a given sentence."""
        if not sentence.strip():
            return ""
            
        # Clean the sentence
        sentence = sentence.strip().strip('.').strip()
        
        # Simple question patterns
        patterns = [
            (r'^(.*?) is (.*?)[.,;]?$', 'What is {}?'),
            (r'^(.*?) are (.*?)[.,;]?$', 'What are {}?'),
            (r'^(.*?) was (.*?)[.,;]?$', 'What was {}?'),
            (r'^(.*?) were (.*?)[.,;]?$', 'What were {}?'),
            (r'^(.*?) can be (.*?)[.,;]?$', 'How can {} be {}?'),
            (r'^(.*?) has (.*?)[.,;]?$', 'What has {}?'),
            (r'^(.*?) have (.*?)[.,;]?$', 'What have {}?'),
        ]
        
        # Try to match patterns
        for pattern, template in patterns:
            match = re.match(pattern, sentence, re.IGNORECASE)
            if match:
                return template.format(match.group(1)).capitalize()
        
        # Default question if no pattern matches
        return f"What is the main point about: {sentence[:50]}...?"
    
    def generate_questions(self, text, num_questions=5):
        """Generate questions from the given text."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        questions = []
        
        for sentence in sentences:
            if len(questions) >= num_questions:
                break
                
            question = self.generate_question(sentence)
            if question and question not in questions:
                questions.append(question)
        
        return questions

# Example usage
if __name__ == "__main__":
    # Create a question generator
    qg = SimpleQuestionGenerator()
    
    # Sample text
    sample_text = """
    Machine learning is a branch of artificial intelligence. 
    It focuses on building systems that learn from data. 
    These systems can improve their performance over time.
    There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.
    Supervised learning uses labeled data to train models.
    Unsupervised learning finds patterns in unlabeled data.
    Reinforcement learning involves training an agent to make decisions through rewards.
    """
    
    # Generate and print questions
    print("Generating questions...\n")
    questions = qg.generate_questions(sample_text, 5)
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
