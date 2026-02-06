# Exam Question Generator

An intelligent Python-based system that automatically generates exam questions from input text using NLP techniques.

## Features

- **Text Preprocessing**: Cleans and preprocesses input text using NLTK
- **Keyword Extraction**: Identifies important concepts using RAKE algorithm
- **Question Generation**: Uses HuggingFace T5 model to generate questions
- **Option Generation**: Creates multiple-choice options with distractors
- **Multiple Interfaces**: CLI and Web interface (Flask) support
- **JSON Output**: Structured output format for easy integration

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Web Interface
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

### Python API
```python
from question_generator import QuestionGenerator

generator = QuestionGenerator()
questions = generator.generate_questions("Your input text here")
```

## Project Structure

- `question_generator.py` - Core question generation logic
- `text_processor.py` - Text cleaning and preprocessing
- `keyword_extractor.py` - Keyword and important sentence extraction
- `option_generator.py` - Multiple choice option generation
- `app.py` - Flask web application
- `