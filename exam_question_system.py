import json
import os
from datetime import datetime
from text_processor import TextProcessor
from keyword_extractor import KeywordExtractor
from question_generator import QuestionGenerator
from option_generator import OptionGenerator
from syllabus_processor import SyllabusProcessor

class ExamQuestionSystem:
    def __init__(self, use_transformers=True):
        """Initialize the complete exam question generation system.
        
        Args:
            use_transformers: Whether to use transformer models for question generation
        """
        print("Initializing Exam Question Generation System...")
        self.text_processor = TextProcessor()
        self.keyword_extractor = KeywordExtractor()
        # Use rule-based generation by default for faster web deployment
        self.question_generator = QuestionGenerator(use_transformers=use_transformers)
        self.option_generator = OptionGenerator()
        self.syllabus_processor = SyllabusProcessor()
        print("System initialized successfully!")
    
    def process_text_file(self, file_path):
        """
        Process a text file and return its content.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: File content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")
    
    def generate_exam_questions(self, input_text, max_questions=5, include_mcq=True, syllabus_text=None):
        """
        Complete pipeline to generate exam questions from input text.
        
        Args:
            input_text (str): Input text to generate questions from
            max_questions (int): Maximum number of questions to generate
            include_mcq (bool): Whether to include multiple choice options
            syllabus_text (str, optional): Syllabus text for topic-based question generation
            
        Returns:
            dict: Generated questions and metadata
        """
        print("Starting question generation pipeline...")
        
        try:
            if not input_text or not input_text.strip():
                raise ValueError("Input text cannot be empty")
                
            # If syllabus text is provided, try syllabus-based generation
            if syllabus_text and syllabus_text.strip():
                syllabus_results = self._generate_syllabus_based_questions(input_text, syllabus_text, max_questions, include_mcq)
                if syllabus_results and syllabus_results.get('questions'):
                    return syllabus_results
                print("Warning: Syllabus-based generation produced no questions. Falling back to standard generation.")
                
            # Otherwise use the standard generation approach
            if not input_text or not input_text.strip():
                raise ValueError("Input text is empty or contains only whitespace")
                
            print(f"Input text length: {len(input_text)} characters")
            
            # Step 1: Text preprocessing
            print("1. Processing and cleaning text...")
            processed_data = self.text_processor.preprocess_text(input_text)
            
            if not processed_data or 'sentences' not in processed_data or not processed_data['sentences']:
                raise ValueError("Failed to process input text into sentences")
            
            print(f"Extracted {len(processed_data['sentences'])} sentences from input")
            
            # Step 2: Extract keywords and important sentences
            print("2. Extracting keywords and important sentences...")
            key_concepts = self.keyword_extractor.extract_key_concepts(
                processed_data['cleaned_text'], 
                processed_data['sentences'],
                top_n_sentences=max(10, max_questions)
            )
            
            if not key_concepts or 'important_sentences' not in key_concepts or not key_concepts['important_sentences']:
                # If no important sentences found, use the first few sentences
                print("Warning: No important sentences found, using first few sentences")
                key_concepts['important_sentences'] = processed_data['sentences'][:max_questions]
            
            print(f"Found {len(key_concepts.get('important_sentences', []))} important sentences")
            
            # Prepare sentences and keywords for generation
            generation_inputs = []
            if key_concepts and 'important_sentences' in key_concepts:
                for item in key_concepts['important_sentences']:
                    if isinstance(item, tuple) and len(item) >= 2:
                        # item is (score, sentence, keyword) or (score, sentence)
                        sentence = item[1]
                        keyword = item[2] if len(item) > 2 else None
                        generation_inputs.append({'context': sentence, 'answer': keyword})
                    elif isinstance(item, str):
                        generation_inputs.append({'context': item, 'answer': None})
            
            # Step 3: Generate questions
            print("3. Generating questions...")
            questions = []
            
            # Generate more questions than requested to ensure we have enough valid ones
            # and to cover all sections (MCQ, Short, Long)
            generation_target = max(max_questions * 2, 10)
            
            try:
                questions = self.question_generator.generate_multiple_questions(
                    generation_inputs, 
                    generation_target
                )
                
                # Ensure we have a list of questions
                if not questions:
                    raise ValueError("No questions were generated")
                    
                # Convert string questions to proper format
                formatted_questions = []
                for i, q in enumerate(questions):
                    if isinstance(q, str):
                        formatted_q = {
                            'question': q,
                            'context': 'Generated from input text',
                            'score': 1.0,
                            'type': 'short_answer',
                            'id': f'q_{i+1}'
                        }
                        formatted_questions.append(formatted_q)
                    elif isinstance(q, dict):
                        # Ensure required fields exist
                        q['question'] = q.get('question', f'Question {i+1}')
                        q['context'] = q.get('context', 'No context provided')
                        q['score'] = q.get('score', 1.0)
                        q['type'] = q.get('type', 'short_answer')
                        q['id'] = q.get('id', f'q_{i+1}')
                        formatted_questions.append(q)
                
                questions = formatted_questions
                
                # Step 4: Generate MCQ options if requested and we have enough questions
                if include_mcq and questions:
                    print("4. Generating multiple choice options...")
                    # Extract global keywords for distractors
                    global_keywords = [k[1] for k in key_concepts.get('keywords', [])]
                    
                    for question_data in questions[:max_questions]:  # Limit to max_questions
                        try:
                            mcq_data = self.option_generator.create_mcq_options(
                                question_data['question'],
                                question_data['context'],
                                correct_answer=question_data.get('correct_answer'),
                                global_keywords=global_keywords
                            )
                            if mcq_data and 'options' in mcq_data and len(mcq_data['options']) >= 2:
                                question_data.update(mcq_data)
                                question_data['type'] = 'mcq'
                                print(f"Generated {len(mcq_data['options'])} options for question")
                            else:
                                print("Not enough options generated, keeping as short answer")
                        except Exception as e:
                            print(f"Error generating MCQ options: {str(e)}"
                                  " (continuing with short answer)")
            
            except Exception as e:
                import traceback
                print(f"Error in question generation: {str(e)}\n{traceback.format_exc()}")
                # Create fallback questions
                questions = [{
                    'question': f"Sample question {i+1} (error: {str(e)[:50]}...)",
                    'context': 'Error occurred during question generation',
                    'score': 0.0,
                    'type': 'error',
                    'id': f'error_{i}'
                } for i in range(min(3, max_questions))]
            
            # Compile results
            results = {
                'metadata': {
                    'input_word_count': processed_data.get('word_count', 0),
                    'input_sentence_count': len(processed_data.get('sentences', [])),
                    'questions_generated': len(questions),
                    'keywords_extracted': len(key_concepts.get('keywords', [])),
                    'named_entities': len(key_concepts.get('named_entities', []))
                },
                'keywords': key_concepts.get('keywords', [])[:10],
                'named_entities': key_concepts.get('named_entities', [])[:10],
                'questions': questions[:max_questions]  # Ensure we don't return more than requested
            }
            
            print(f"Successfully generated {len(results['questions'])} questions")
            return results
            
        except Exception as e:
            import traceback
            error_msg = f"Error in generate_exam_questions: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Return a minimal response with error information
            return {
                'metadata': {
                    'error': str(e),
                    'input_length': len(input_text) if input_text else 0,
                    'questions_generated': 0
                },
                'keywords': [],
                'named_entities': [],
                'questions': [{
                    'question': f"Error generating questions: {str(e)[:100]}",
                    'context': 'An error occurred during question generation',
                    'score': 0.0,
                    'type': 'error',
                    'id': 'error_0'
                }]
            }
        
        print(f"✅ Generated {len(questions)} questions successfully!")
        return results
    
    def save_questions_to_json(self, questions_data, output_file):
        """
        Save generated questions to a JSON file.
        
        Args:
            questions_data (dict): Generated questions data
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(questions_data, file, indent=2, ensure_ascii=False)
            print(f"✅ Questions saved to {output_file}")
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
    
    def display_questions_console(self, questions_data):
        """
        Display generated questions in a formatted console output.
        
        Args:
            questions_data (dict): Generated questions data
        """
        print("\n" + "="*80)
        print("GENERATED EXAM QUESTIONS")
        print("="*80)
        
        # Display metadata
        metadata = questions_data['metadata']
        print(f"\n📊 STATISTICS:")
        print(f"   • Input text: {metadata['input_word_count']} words, {metadata['input_sentence_count']} sentences")
        print(f"   • Keywords extracted: {metadata['keywords_extracted']}")
        print(f"   • Named entities found: {metadata['named_entities']}")
        print(f"   • Questions generated: {metadata['questions_generated']}")
        
        # Display top keywords
        print(f"\n🔑 TOP KEYWORDS:")
        for score, keyword in questions_data['keywords'][:5]:
            print(f"   • {keyword} (score: {score:.2f})")
        
        # Display questions
        print(f"\n❓ QUESTIONS:")
        for i, q in enumerate(questions_data['questions'], 1):
            print(f"\n{i}. {q['question']}")
            
            if 'options' in q:
                print("   Options:")
                for j, option in enumerate(q['options'], 1):
                    marker = "✓" if j-1 == q['correct_index'] else " "
                    print(f"   {marker} {chr(64+j)}. {option}")
            
            print(f"   Context: {q['context'][:100]}...")
            print(f"   Confidence: {q['score']:.2f}")
        
        print("\n" + "="*80)

    def _generate_syllabus_based_questions(self, content_text, syllabus_text, max_questions=10, include_mcq=True):
        """
        Generate questions based on syllabus topics.
        
        Args:
            content_text (str): The content text to generate questions from
            syllabus_text (str): The syllabus text with units and topics
            max_questions (int): Maximum number of questions to generate
            include_mcq (bool): Whether to include multiple choice options
            
        Returns:
            dict: Generated questions and metadata
        """
        print("Generating syllabus-based questions...")
        
        try:
            # Generate questions by topic
            questions_by_topic = self.syllabus_processor.generate_topic_based_questions(
                syllabus_text=syllabus_text,
                content_text=content_text,
                questions_per_topic=3  # Will be adjusted based on max_questions
            )
            
            # Flatten questions from all topics
            all_questions = []
            for topic, questions in questions_by_topic.items():
                for q in questions:
                    q['topic'] = topic
                    all_questions.append(q)
            
            # Limit to max_questions
            all_questions = all_questions[:max_questions]
            
            # Generate options for MCQs if needed
            if include_mcq:
                for question in all_questions:
                    if 'options' not in question and 'context' in question:
                        try:
                            mcq_data = self.option_generator.create_mcq_options(
                                question['question'],
                                question['context'],
                                num_options=4
                            )
                            if mcq_data and 'options' in mcq_data and len(mcq_data['options']) >= 2:
                                question.update(mcq_data)
                        except Exception as e:
                            print(f"Error generating options: {e}")
            
            # Prepare results
            results = {
                'metadata': {
                    'total_questions': len(all_questions),
                    'topics_covered': list(questions_by_topic.keys()),
                    'generated_at': str(datetime.now())
                },
                'questions': all_questions
            }
            
            return results
            
        except Exception as e:
            print(f"Error in syllabus-based question generation: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that work and react like humans. Machine learning is a subset of AI that provides systems the ability 
    to automatically learn and improve from experience without being explicitly programmed. Deep learning 
    is a subset of machine learning that uses neural networks with three or more layers. These neural 
    networks attempt to simulate the behavior of the human brain to learn from large amounts of data. 
    Python is one of the most popular programming languages for AI development due to its simplicity 
    and extensive libraries like TensorFlow and PyTorch.
    """
    
    try:
        # Initialize system
        system = ExamQuestionSystem()
        
        # Generate questions
        results = system.generate_exam_questions(sample_text, max_questions=3)
        
        # Display results
        system.display_questions_console(results)
        
        # Save to JSON
        system.save_questions_to_json(results, "sample_questions.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
