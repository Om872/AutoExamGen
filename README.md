---
title: AutoExamGen
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# AutoExamGen – AI Based Exam Question Generator

AutoExamGen is an intelligent Python-based system that automatically generates exam questions
(MCQ, short answer, and long answer) from input text using Natural Language Processing (NLP) techniques.

## Features

- **Text Preprocessing**: Cleans and preprocesses input text using NLTK
- **Keyword Extraction**: Identifies important concepts using RAKE algorithm
- **Question Generation**: Uses HuggingFace T5 model to generate questions
- **Option Generation**: Creates multiple-choice options with distractors
- **Syllabus-Based Generation**: Generate questions topic-wise from syllabus input
- **Download Support**: Export generated question papers
- **Multiple Interfaces**: CLI and Web interface (Flask) support
- **JSON Output**: Structured output format for easy integration

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data (run once):
```bash
python setup_nltk.py

## Usage

### Web Interface
The web interface allows users to upload text or documents and configure question types and difficulty.

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

- `app.py` – Flask web application (main entry point)
- `exam_question_system.py` – Orchestrates the complete question generation pipeline
- `question_generator.py` – Core question generation logic
- `text_processor.py` – Text cleaning and preprocessing
- `keyword_extractor.py` – Keyword and concept extraction (RAKE)
- `option_generator.py` – MCQ option and distractor generation
- `syllabus_processor.py` – Syllabus-based question generation
- `local_question_generator.py` – Transformer-based (T5) question generator
- `setup_nltk.py` – NLTK data setup script

## Technologies Used

- Python
- Flask
- Natural Language Processing (NLP)
- NLTK
- RAKE Algorithm
- HuggingFace Transformers (T5)
- HTML, CSS (Web Interface)
## Author


**Om Namdev**  
B.Tech AI & DS  
Aspiring Data Scientist
