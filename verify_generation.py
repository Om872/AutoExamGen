import logging
from exam_question_system import ExamQuestionSystem
from option_generator import OptionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)

def verify_generation():
    # Mock content
    content = """
    Data visualization is the graphical representation of information and data. 
    By using visual elements like charts, graphs, and maps, data visualization 
    tools provide an accessible way to see and understand trends, outliers, and 
    patterns in data. In the world of Big Data, data visualization tools and 
    technologies are essential to analyze massive amounts of information and 
    make data-driven decisions.
    
    Design principles in data visualization include understanding the audience, 
    choosing the right chart type, and using color effectively. Good design 
    makes complex data more accessible, understandable, and usable.
    
    Exploratory Data Analysis (EDA) is an approach to analyzing data sets to 
    summarize their main characteristics, often with visual methods. A statistical 
    model can be used or not, but primarily EDA is for seeing what the data can 
    tell us beyond the formal modeling or hypothesis testing task.
    """
    
    print("Initializing ExamQuestionSystem...")
    system = ExamQuestionSystem(use_transformers=False) # Use rule-based for speed in test
    
    print("\nGenerating questions (Target: 2 MCQ, 1 Short, 1 Long)...")
    # We simulate the logic in app.py
    num_mcq = 2
    num_short = 1
    num_long = 1
    total_needed = num_mcq + num_short + num_long
    
    results = system.generate_exam_questions(
        input_text=content,
        max_questions=total_needed,
        include_mcq=False,
        syllabus_text=content
    )
    
    all_questions = results.get('questions', [])
    print(f"\nTotal questions generated: {len(all_questions)}")
    
    # Simulate app.py distribution logic
    generated_questions = {
        'mcq_questions': [],
        'short_questions': [],
        'long_questions': []
    }
    
    # Filter out questions that are too simple for Long answers
    long_candidates = [q for q in all_questions if len(q.get('context', '').split()) > 10]
    short_candidates = [q for q in all_questions if q not in long_candidates]
    
    # If we don't have enough long candidates, take from short
    if len(long_candidates) < num_long:
        needed = num_long - len(long_candidates)
        long_candidates.extend(short_candidates[:needed])
        short_candidates = short_candidates[needed:]
    
    # 3. Process Long Questions
    for _ in range(num_long):
        if long_candidates:
            q = long_candidates.pop(0)
            q['type'] = 'long_answer'
            generated_questions['long_questions'].append(q)
            if q in all_questions:
                all_questions.remove(q)
    
    # 2. Process Short Questions
    for _ in range(num_short):
        if short_candidates:
            q = short_candidates.pop(0)
            q['type'] = 'short_answer'
            generated_questions['short_questions'].append(q)
            if q in all_questions:
                all_questions.remove(q)
        elif all_questions:
            q = all_questions.pop(0)
            q['type'] = 'short_answer'
            generated_questions['short_questions'].append(q)

    # 1. Process MCQs
    global_keywords = [k[1] for k in results.get('keywords', [])]
    print(f"\nGlobal Keywords (Cleaned): {global_keywords}")
    
    for _ in range(num_mcq):
        if all_questions:
            q = all_questions.pop(0)
            try:
                mcq_data = system.option_generator.create_mcq_options(
                    q['question'],
                    q['context'],
                    correct_answer=q.get('correct_answer'),
                    global_keywords=global_keywords
                )
                if mcq_data and 'options' in mcq_data:
                    q.update(mcq_data)
                    q['type'] = 'mcq'
                    generated_questions['mcq_questions'].append(q)
            except Exception as e:
                print(f"Error generating options: {e}")

    # Print Results
    print("\n--- Generation Results ---")
    print(f"MCQs: {len(generated_questions['mcq_questions'])}")
    for q in generated_questions['mcq_questions']:
        print(f"  Q: {q['question']}")
        print(f"  Options: {q.get('options')}")
        
    print(f"\nShort Questions: {len(generated_questions['short_questions'])}")
    for q in generated_questions['short_questions']:
        print(f"  Q: {q['question']}")
        
    print(f"\nLong Questions: {len(generated_questions['long_questions'])}")
    for q in generated_questions['long_questions']:
        print(f"  Q: {q['question']}")

if __name__ == "__main__":
    verify_generation()
