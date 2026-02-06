#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test to verify all modules can be imported and basic functionality works."""

import sys

def test_imports():
    """Test if all modules can be imported."""
    print("Testing module imports...")
    
    try:
        print("  [OK] Importing text_processor...")
        from text_processor import TextProcessor
        tp = TextProcessor()
        print("    [OK] TextProcessor initialized")
        
        print("  [OK] Importing keyword_extractor...")
        from keyword_extractor import KeywordExtractor
        ke = KeywordExtractor()
        print("    [OK] KeywordExtractor initialized")
        
        print("  [OK] Importing question_generator...")
        from question_generator import QuestionGenerator
        qg = QuestionGenerator(use_transformers=False)
        print("    [OK] QuestionGenerator initialized")
        
        print("  [OK] Importing option_generator...")
        from option_generator import OptionGenerator
        og = OptionGenerator()
        print("    [OK] OptionGenerator initialized")
        
        print("  [OK] Importing syllabus_processor...")
        from syllabus_processor import SyllabusProcessor
        sp = SyllabusProcessor()
        print("    [OK] SyllabusProcessor initialized")
        
        print("  [OK] Importing exam_question_system...")
        from exam_question_system import ExamQuestionSystem
        print("    [OK] ExamQuestionSystem imported")
        
        print("  [OK] Importing app...")
        from app import app
        print("    [OK] Flask app imported")
        
        print("\n[SUCCESS] All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"\n[WARNING] Error during initialization: {e}")
        print("   (This might be due to missing NLTK data)")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from text_processor import TextProcessor
        from keyword_extractor import KeywordExtractor
        
        tp = TextProcessor()
        ke = KeywordExtractor()
        
        test_text = "Python is a programming language. It is widely used for web development and data science."
        
        print("  [OK] Testing text preprocessing...")
        processed = tp.preprocess_text(test_text)
        assert 'sentences' in processed
        assert len(processed['sentences']) > 0
        print(f"    [OK] Processed {len(processed['sentences'])} sentences")
        
        print("  [OK] Testing keyword extraction...")
        keywords = ke.extract_keywords_rake(test_text, max_keywords=5)
        assert len(keywords) > 0
        print(f"    [OK] Extracted {len(keywords)} keywords")
        
        print("\n[SUCCESS] Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n[WARNING] Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Project Import and Basic Functionality Test")
    print("=" * 60)
    
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "=" * 60)
            print("[SUCCESS] PROJECT IS WORKING!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("[WARNING] Imports work but functionality may have issues")
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("[ERROR] PROJECT HAS IMPORT ERRORS")
        print("=" * 60)
        sys.exit(1)
