from local_question_generator import LocalQuestionGenerator

def test_local_generator():
    print("Testing Local Question Generator...")
    
    # Sample text
    sample_text = """
    The water cycle describes how water evaporates from the Earth's surface, 
    rises into the atmosphere, cools and condenses into rain or snow in clouds, 
    and falls again to the surface as precipitation.
    """
    
    # Initialize the generator
    print("Initializing model (this may take a minute)...")
    qg = LocalQuestionGenerator()
    
    # Generate questions
    print("\nGenerating questions...")
    questions = qg.generate_questions(sample_text, num_questions=3)
    
    # Print results
    print("\nGenerated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

if __name__ == "__main__":
    test_local_generator()
