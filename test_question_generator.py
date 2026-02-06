from question_generator import QuestionGenerator

def test_question_generator():
    print("Testing Question Generator...")
    
    # Initialize the question generator
    qg = QuestionGenerator(use_transformers=False)  # Using rule-based for faster testing
    
    # Sample text about machine learning
    sample_text = """
    Machine learning is a branch of artificial intelligence that focuses on building systems 
    that learn from data. These systems can improve their performance over time without being 
    explicitly programmed. There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Supervised learning uses labeled data 
    to train models, while unsupervised learning finds patterns in unlabeled data. 
    Reinforcement learning involves training an agent to make decisions by rewarding desired 
    behaviors and/or punishing undesired ones.
    """
    
    print("\nGenerating questions...")
    questions = qg.generate_questions(sample_text, num_questions=3)
    
    print("\nGenerated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

if __name__ == "__main__":
    test_question_generator()
