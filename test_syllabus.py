import logging
import sys
import os

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

from syllabus_processor import SyllabusProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_syllabus_parsing():
    syllabus_text = """Unit 1.0 Introduction (9 Lectures) Self-Learning (SL) 1.1 Data for Graphics. 1.2 Design principles 1.3 Value for visualization 1.4 Categorical 1.5 Time series 1.6 statistical data graphics I 1.7 statistical data graphics II 1.8 Introduction to Visualization Tools I 1.9 Introduction to Visualization Tools II 
Unit 2.0 Graphics Pipeline and Aesthetics and Perception (10 Lectures) 2.1 Primitives: vertices edges and triangles 2.2 Model transforms 2.3 Translations 2.4 Rotations 2.5 scaling 2.6 View transform, Perspective transform, window transform 2.7 Graphical Perception Theory 2.8 Experimentation, and the Application 2.9 Graphical Integrity, Layering and Separation 2.10 Color and Information, Using Space"""
    
    print("Initializing SyllabusProcessor...")
    try:
        processor = SyllabusProcessor()
        
        print("\nParsing syllabus...")
        units = processor.parse_syllabus(syllabus_text)
        
        print("\nParsing Results:")
        for unit, topics in units.items():
            print(f"\nUnit: {unit}")
            print(f"Topics found: {len(topics)}")
            for topic in topics:
                print(f"  - {topic}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_syllabus_parsing()
