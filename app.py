from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
import os
import json
import tempfile
from werkzeug.utils import secure_filename
from exam_question_system import ExamQuestionSystem
from datetime import datetime
import uuid
import threading
import time
from docx import Document
import PyPDF2
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_INPUT_FOLDER'] = os.path.join(tempfile.gettempdir(), 'eqg_inputs')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.secret_key = 'your-secret-key-change-this-in-production'

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_INPUT_FOLDER'], exist_ok=True)

# Global variables for question system
question_system = None
system_loading = False
system_load_error = None

def read_file_content(filepath):
    """Read content from a file based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.pdf':
            text = ""
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        elif ext == '.docx':
            doc = Document(filepath)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        else:
            # Default to text file
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
                
    except Exception as e:
        raise Exception(f"Error reading {ext} file: {str(e)}")

def get_question_system():
    """Get or initialize the question generation system."""
    global question_system, system_loading, system_load_error
    
    if question_system is None and not system_loading:
        if system_load_error:
            raise Exception(f"System failed to load: {system_load_error}")
        
        system_loading = True
        try:
            print("Initializing question generation system...")
            question_system = ExamQuestionSystem()
            print("Question generation system loaded successfully!")
        except Exception as e:
            system_load_error = str(e)
            system_loading = False
            raise e
        finally:
            system_loading = False
    
    if system_loading:
        raise Exception("System is still loading, please wait...")
    
    return question_system

# Utility: parse human-readable duration text into minutes when possible
def parse_duration_to_minutes(duration_text):
    try:
        if not duration_text:
            return None
        text = duration_text.strip().lower()
        # Normalize
        text = text.replace('hrs', 'h').replace('hr', 'h').replace('hours', 'h').replace('hour', 'h')
        text = text.replace('minutes', 'm').replace('minute', 'm').replace('mins', 'm').replace('min', 'm')
        # Patterns like '2h 30m'
        import re
        hours = 0
        minutes = 0
        # Match hours
        h_match = re.search(r"(\d+)\s*h", text)
        if h_match:
            hours = int(h_match.group(1))
        # Match minutes
        m_match = re.search(r"(\d+)\s*m", text)
        if m_match:
            minutes = int(m_match.group(1))
        if h_match or m_match:
            return hours * 60 + minutes
        # If only a number, treat as minutes
        just_num = re.fullmatch(r"\s*(\d+)\s*", duration_text)
        if just_num:
            return int(just_num.group(1))
        # If something like '3 hour' without m, captured above; if unparseable, return None
        return None
    except Exception:
        return None

@app.route('/')
def index():
    """Welcome page before the step flow."""
    # Clear any existing session data
    session.clear()
    return render_template('welcome.html', project_name='AutoExamGen')

@app.route('/step1', methods=['GET'])
def step1_input():
    """Step 1: Syllabus input page."""
    return render_template('step1_input.html')

@app.route('/step2', methods=['GET', 'POST'])
def step2_configuration():
    """Step 2: Question configuration page."""
    if request.method == 'GET':
        # If user tries to access /step2 directly, redirect to step1
        return redirect(url_for('step1_input'))
    
    # Handle POST request (form submission from step1)
    content = None
    
    try:
        # Get text input (from textarea)
        text_input = request.form.get('text_input', '').strip()
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                try:
                    # Save the uploaded file
                    filename = secure_filename(file.filename)
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        os.makedirs(app.config['UPLOAD_FOLDER'])
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Store file path in session
                    session['content_file'] = filepath
                    session.pop('content_text', None)  # Clear any text content if it exists
                    
                    # Read the file content for processing
                    content = read_file_content(filepath)
                    print(f"File uploaded successfully: {filename}, Content length: {len(content)}")
                    
                except Exception as e:
                    error_msg = f'Error processing file: {str(e)}'
                    print(error_msg)
                    flash(error_msg, 'error')
                    return redirect(url_for('step1_input'))
        
        # If no file but text content is provided
        if not content and text_input:
            # For small text content, store directly in session
            if len(text_input) < 2000:
                session['content_text'] = text_input
                content = text_input
            else:
                # For large content, save to a temporary file
                temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{int(time.time())}.txt')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(text_input)
                session['content_file'] = temp_file
                content = text_input
            print(f"Text input processed, Content length: {len(content)}")
        
        # Validate that we have content
        if not content or not content.strip():
            flash('Please provide either a file or paste content', 'error')
            return redirect(url_for('step1_input'))
        
        # Process the content for question generation
        try:
            # Initialize question system if not already done
            print("Initializing question system...")
            question_system = get_question_system()
            print("Question system initialized successfully")
            
            # Store word count for progress display
            word_count = len(content.split())
            session['word_count'] = word_count
            print(f"Content processed: {word_count} words")
            
            # Render the configuration page
            return render_template('step2_config.html', 
                                 word_count=word_count,
                                 has_syllabus=bool(session.get('syllabus_text', '')))
            
        except Exception as e:
            error_msg = f'Error initializing question system: {str(e)}'
            print(error_msg)
            import traceback
            traceback.print_exc()
            flash(error_msg, 'error')
            return redirect(url_for('step1_input'))
            
    except Exception as e:
        error_msg = f'An error occurred: {str(e)}'
        print(error_msg)
        import traceback
        traceback.print_exc()
        flash(error_msg, 'error')
        return redirect(url_for('step1_input'))

@app.route('/generate', methods=['POST'])
def step3_generate():
    """Step 3: Generate and display question paper."""
    try:
        # Get form data
        num_questions = int(request.form.get('num_questions', 5))
        question_types = request.form.getlist('question_types')
        
        # Get content from session or uploaded file
        content = None
        
        # Check for uploaded file first
        if 'content_file' in session and os.path.exists(session['content_file']):
            try:
                content = read_file_content(session['content_file'])
            except Exception as e:
                flash(f'Error reading uploaded file: {str(e)}', 'error')
                return redirect(url_for('step1_input'))
        # Check for direct text content
        elif 'content_text' in session and session['content_text']:
            content = session['content_text']
        
        # If no content found, redirect to step 1
        if not content:
            flash('No content found. Please provide content first.', 'error')
            return redirect(url_for('step1_input'))
            
        # Initialize question system
        try:
            question_system = get_question_system()
            
            # Helper function to safely get integer values from form
            def get_int(form, key, default=0):
                try:
                    return int(form.get(key, default))
                except (ValueError, TypeError):
                    return default
            
            # Store configuration in session with all required fields and safe defaults
            config = {
                'exam_name': request.form.get('exam_name', 'Sample Exam'),
                'subject_name': request.form.get('subject_name', 'Subject'),
                'duration': get_int(request.form, 'duration', 60),
                'short_questions': get_int(request.form, 'short_questions', 2),
                'short_marks': get_int(request.form, 'short_marks', 2),
                'long_questions': get_int(request.form, 'long_questions', 1),
                'long_marks': get_int(request.form, 'long_marks', 5),
                'long_attempt': get_int(request.form, 'long_attempt', 1),
                'mcq_questions': get_int(request.form, 'mcq_questions', 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            session['exam_config'] = config
            
            # Get the syllabus/content text
            content = ""
            if 'content_text' in session:
                content = session['content_text']
            elif 'content_file' in session and os.path.exists(session['content_file']):
                content = read_file_content(session['content_file'])
            
            if not content.strip():
                raise ValueError("No content available for question generation.")
            
            # Generate questions based on content and configuration
            print("Generating questions from content...")
            
            # Get number of questions for each type
            num_mcq = config['mcq_questions']
            num_short = config['short_questions']
            num_long = config['long_questions']
            
            # Generate questions using the question system
            # We request enough questions for all sections, but without auto-MCQ generation
            # so we can handle categorization manually
            total_questions_needed = num_mcq + num_short + num_long
            
            results = question_system.generate_exam_questions(
                input_text=content,
                max_questions=total_questions_needed,
                include_mcq=False,  # We'll generate options manually for specific questions
                syllabus_text=content
            )
            
            all_questions = results.get('questions', [])
            
            # Initialize categories
            generated_questions = {
                'mcq_questions': [],
                'short_questions': [],
                'long_questions': []
            }
            
            # Distribute questions
            # We prioritize Short and Long questions first to ensure they get content, 
            # as MCQs are easier to fallback/generate
            
            # Filter out questions that are too simple for Long answers
            long_candidates = [q for q in all_questions if len(q.get('context', '').split()) > 10]
            short_candidates = [q for q in all_questions if q not in long_candidates]
            
            # If we don't have enough long candidates, take from short
            if len(long_candidates) < num_long:
                needed = num_long - len(long_candidates)
                long_candidates.extend(short_candidates[:needed])
                short_candidates = short_candidates[needed:]
            
            # 3. Process Long Questions (Prioritize these)
            for _ in range(num_long):
                if long_candidates:
                    q = long_candidates.pop(0)
                    q['type'] = 'long_answer'
                    generated_questions['long_questions'].append(q)
                    # Remove from all_questions so we don't reuse
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
                elif all_questions: # Fallback to any remaining
                    q = all_questions.pop(0)
                    q['type'] = 'short_answer'
                    generated_questions['short_questions'].append(q)

            # 1. Process MCQs (Use remaining questions)
            # Extract global keywords for distractors
            global_keywords = [k[1] for k in results.get('keywords', [])]
            
            for _ in range(num_mcq):
                if all_questions:
                    q = all_questions.pop(0)
                    # Generate options for this question
                    try:
                        mcq_data = question_system.option_generator.create_mcq_options(
                            q['question'],
                            q['context'],
                            correct_answer=q.get('correct_answer'),
                            global_keywords=global_keywords
                        )
                        if mcq_data and 'options' in mcq_data:
                            q.update(mcq_data)
                            q['type'] = 'mcq'
                            generated_questions['mcq_questions'].append(q)
                        else:
                            # Fallback if option generation fails
                            q['type'] = 'short_answer'
                            generated_questions['short_questions'].append(q)
                    except Exception as e:
                        print(f"Error generating options: {e}")
                        q['type'] = 'short_answer'
                        generated_questions['short_questions'].append(q)
            
            # Store the generated questions
            session['generated_questions'] = generated_questions
            
            # Calculate and store total marks
            total_marks = (
                (len(session['generated_questions']['short_questions']) * config['short_marks']) +
                (len(session['generated_questions']['long_questions']) * config['long_marks']) +
                len(session['generated_questions']['mcq_questions'])  # 1 mark per MCQ
            )
            session['total_marks'] = total_marks
            
            # Redirect to results page
            return redirect(url_for('show_results'))
            
        except Exception as e:
            error_msg = f'Error generating questions: {str(e)}'
            print(error_msg)
            flash(error_msg, 'error')
            return redirect(url_for('step1_input'))
            
    except Exception as e:
        error_msg = f'An error occurred: {str(e)}'
        print(error_msg)
        flash(error_msg, 'error')
        return redirect(url_for('step1_input'))

@app.route('/download')
def download_paper():
    """Download the generated question paper as HTML."""
    try:
        if 'question_paper' not in session:
            return redirect(url_for('index'))
        
        # Generate a unique filename
        filename = f"question_paper_{uuid.uuid4().hex[:8]}.html"
        
        # Render the printable version
        # Determine display duration similarly to step3
        cfg = session['config']
        display_duration = cfg.get('exam_duration') if cfg.get('exam_duration') else int(round(session['total_marks'] * 1.5))

        html_content = render_template('printable_paper.html',
                                     question_paper=session['question_paper'],
                                     config=cfg,
                                     total_marks=session['total_marks'],
                                     exam_date=session['exam_date'],
                                     display_duration=display_duration)
        
        # Create a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(temp_file, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': f'Error downloading paper: {str(e)}'}), 500

@app.route('/results')
def show_results():
    """Display the generated questions."""
    if 'generated_questions' not in session or 'exam_config' not in session:
        flash('No questions generated yet. Please start from the beginning.', 'error')
        return redirect(url_for('step1_input'))
    
    # Get config with defaults
    config = session.get('exam_config', {})
    questions = session.get('generated_questions', {})
    
    # Ensure all required question types exist in the questions dictionary
    for qtype in ['mcq_questions', 'short_questions', 'long_questions']:
        if qtype not in questions:
            questions[qtype] = []
    
    # Calculate total marks
    total_marks = 0
    if 'mcq_questions' in questions:
        total_marks += len(questions['mcq_questions'])
    if 'short_questions' in questions:
        total_marks += len(questions['short_questions']) * 2  # 2 marks per short question
    if 'long_questions' in questions:
        total_marks += len(questions['long_questions']) * 5  # 5 marks per long question
    
    # Get exam date from config or use current date
    exam_date = config.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return render_template('step3_result.html',
                         question_paper=questions,  # Changed from questions to question_paper
                         config=config,
                         total_marks=total_marks,
                         exam_date=datetime.strptime(exam_date, '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y'),
                         display_duration=config.get('duration', 60))  # Use configured duration or default to 60 minutes

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'Exam Question Generator'})

@app.route('/api/warmup')
def warmup():
    """Warmup endpoint to initialize the system."""
    try:
        system = get_question_system()
        return jsonify({
            'status': 'ready', 
            'message': 'Question generation system is ready',
            'uses_transformers': system.question_generator.use_transformers
        })
    except Exception as e:
        return jsonify({
            'status': 'loading' if 'still loading' in str(e) else 'error',
            'message': str(e)
        }), 202 if 'still loading' in str(e) else 500

if __name__ == '__main__':
    print("🌐 Starting Flask Web Application...")
    print("📝 Exam Question Generator Web Interface")
    print("🔗 Access the application at: http://localhost:5000")
    print("💡 Using rule-based question generation for faster startup")
    print("⚡ System will initialize on first use")

    # Configure Flask for better timeout handling
    app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
