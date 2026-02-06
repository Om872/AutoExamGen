from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LocalQuestionGenerator:
    def __init__(self):
        print("Loading local question generation model...")
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        
        # Load the model and tokenizer
        model_name = "valhalla/t5-base-qa-qg-hl"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Initialize the pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        print("Model loaded successfully!")
    
    def generate_questions(self, text, num_questions=5, max_length=64):
        """Generate questions from the given text."""
        if not text.strip():
            return []
            
        try:
            # Prepare the input text
            input_text = f"generate questions: {text}"
            
            # Generate questions
            results = self.generator(
                input_text,
                max_length=max_length,
                num_return_sequences=num_questions,
                num_beams=5,
                early_stopping=True
            )
            
            # Extract and clean the generated questions
            questions = [result['generated_text'].strip() for result in results]
            return questions
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize the generator
    qg = LocalQuestionGenerator()
    
    # Sample text
    sample_text = """
    Machine learning is a branch of artificial intelligence that focuses on building systems 
    that learn from data. These systems can improve their performance over time without being 
    explicitly programmed. There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning.
    """
    
    # Generate questions
    print("\nGenerating questions...")
    questions = qg.generate_questions(sample_text, num_questions=3)
    
    # Print the results
    print("\nGenerated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
