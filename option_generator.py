import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

class OptionGenerator:
    def __init__(self):
        """Initialize the option generator with NLTK resources."""
        try:
            # Download required NLTK data with explicit resource names
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('universal_tagset', quiet=True)
            nltk.download('tagsets', quiet=True)
            
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
            self.word_net_lemmatizer = nltk.WordNetLemmatizer()
            
            # POS tag mapping for WordNet
            self.pos_mapping = {
                'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
                'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
                'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
                'RB': 'r', 'RBR': 'r', 'RBS': 'r'
            }
            
        except Exception as e:
            print(f"Error initializing OptionGenerator: {str(e)}")
            raise
        
    def _get_synonyms(self, word, pos=None):
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        # Skip if word is too short or a stop word
        if len(word) < 3 or word.lower() in self.stop_words:
            return []
            
        try:
            wordnet_pos = self.pos_mapping.get(pos, None) if pos else None
            
            # Try with the provided POS tag first
            if wordnet_pos:
                for syn in wordnet.synsets(word, pos=wordnet_pos):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym != word and len(synonym.split()) == 1:
                            synonyms.add(synonym)
                            
            # If no synonyms found, try without POS tag
            if not synonyms:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym != word and len(synonym.split()) == 1:
                            synonyms.add(synonym)
            
            # If still no synonyms, try with lemmatization
            if not synonyms and pos and pos.startswith('VB'):
                lemma = self.word_net_lemmatizer.lemmatize(word, pos='v')
                if lemma != word:
                    for syn in wordnet.synsets(lemma, pos='v'):
                        for l in syn.lemmas():
                            synonym = l.name().replace('_', ' ').lower()
                            if synonym != word and len(synonym.split()) == 1:
                                synonyms.add(synonym)
                                
        except Exception as e:
            print(f"Error getting synonyms for '{word}': {str(e)}")
        
        return list(synonyms)[:10]  # Return at most 10 synonyms
    
    def _get_distractors(self, word, pos=None, num=3):
        """Generate distractors for a given word."""
        distractors = set()
        
        try:
            # Get synonyms first
            synonyms = self._get_synonyms(word, pos)
            distractors.update(synonyms[:num])
            
            # If not enough synonyms, add similar words
            if len(distractors) < num:
                wordnet_pos = self.pos_mapping.get(pos, None) if pos else None
                similar_words = []
                
                for syn in wordnet.synsets(word, pos=wordnet_pos):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            similar_words.append(lemma.name().replace('_', ' ').lower())
                
                # Add similar words that aren't already in distractors
                for w in similar_words:
                    if w not in distractors and w != word:
                        distractors.add(w)
                        if len(distractors) >= num:
                            break
        except Exception as e:
            print(f"Error generating distractors for '{word}': {str(e)}")
        
        return list(distractors)[:num]
    
    def extract_answer_from_context(self, question, context):
        """
        Extract the most likely answer from the context based on the question.
        This version uses simple string matching instead of POS tagging.
        
        Args:
            question (str): Generated question
            context (str): Source sentence/context
            
        Returns:
            str: Extracted answer
        """
        try:
            q_lower = question.lower()
            context_lower = context.lower()
            
            # Common patterns for answers
            patterns = [
                ('what is', 'is'),
                ('what are', 'are'),
                ('what was', 'was'),
                ('what were', 'were'),
                ('who is', 'is'),
                ('who are', 'are'),
                ('who was', 'was'),
                ('who were', 'were'),
                ('where is', 'is'),
                ('where are', 'are'),
                ('when is', 'is'),
                ('when was', 'was')
            ]
            
            # Try to find a direct answer using common patterns
            for q_pattern, verb in patterns:
                if q_lower.startswith(q_pattern):
                    # Look for the pattern "[verb] [answer]" in the context
                    verb_pos = context_lower.find(verb)
                    if verb_pos != -1:
                        # Get the text after the verb
                        answer_part = context[verb_pos + len(verb):].strip(' ,.?!')
                        # Return the first word or phrase
                        return answer_part.split(',')[0].split('.')[0].strip()
            
            # Fallback: return the first proper noun or capitalized word not in the question
            words = context.split()
            for word in words:
                # Skip short words and words that are in the question
                if (len(word) > 2 and word[0].isupper() and 
                    word.lower() not in q_lower and 
                    word.lower() not in self.stop_words):
                    return word.strip(',.!?;:')
            
            # Last resort: return the first noun-like word
            for word in words:
                if len(word) > 3 and word.lower() not in q_lower and word.lower() not in self.stop_words:
                    return word.strip(',.!?;:')
            
            # If all else fails, return the first word that's not a stop word
            for word in words:
                if word.lower() not in self.stop_words and len(word) > 2:
                    return word.strip(',.!?;:')
            
            # Final fallback
            return context.split()[0] if context else "Unknown"
            
        except Exception as e:
            print(f"Error extracting answer: {str(e)}")
            # Return the first word as fallback
            return context.split()[0] if context else "Unknown"
    
    def create_mcq_options(self, question, context, num_options=4, correct_answer=None, global_keywords=None):
        """
        Create multiple choice options for a given question and context.
        
        Args:
            question (str): The question text
            context (str): The context from which the question was generated
            num_options (int): Number of options to generate (including correct answer)
            correct_answer (str, optional): The correct answer if known
            global_keywords (list, optional): List of keywords from the entire document to use as distractors
            
        Returns:
            dict: Dictionary containing options and correct index
        """
        try:
            # Extract the correct answer from context if not provided
            if not correct_answer:
                correct_answer = self.extract_answer_from_context(question, context)
            
            # If we couldn't extract a good answer, use a fallback
            if not correct_answer or correct_answer == "Unknown":
                return {
                    'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                    'correct_index': 0,
                    'correct_answer': 'Option A'
                }
            
            # Generate distractors
            distractors = self._get_distractors(
                correct_answer,
                num=min(10, num_options * 2)  # Generate more than needed to filter
            )
            
            # Ensure we have unique distractors
            distractors = list(set(d for d in distractors if d.lower() != correct_answer.lower()))
            
            # If we don't have enough distractors, try using global keywords
            if len(distractors) < num_options - 1 and global_keywords:
                # Filter keywords to ensure they are not the correct answer
                potential_distractors = [k for k in global_keywords if k.lower() != correct_answer.lower()]
                # Shuffle to get random ones
                random.shuffle(potential_distractors)
                
                for kw in potential_distractors:
                    if kw not in distractors:
                        distractors.append(kw)
                        if len(distractors) >= num_options + 2:  # Get a few extra
                            break
            
            # If we still don't have enough distractors, add some generic ones
            generic_distractors = [
                'True', 'False', 'Yes', 'No', 'Maybe', 'Always', 'Never',
                'Sometimes', 'Often', 'Rarely', 'All of the above', 'None of the above'
            ]
            
            while len(distractors) < num_options - 1 and generic_distractors:
                distractor = generic_distractors.pop(0)
                if distractor.lower() != correct_answer.lower() and distractor not in distractors:
                    distractors.append(distractor)
            
            # Select the final set of options
            options = [correct_answer] + distractors[:(num_options-1)]
            random.shuffle(options)
            
            # Find the index of the correct answer
            correct_index = options.index(correct_answer) if correct_answer in options else 0
            
            return {
                'options': options,
                'correct_index': correct_index,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            print(f"Error generating options: {str(e)}")
            # Fallback options
            return {
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'correct_index': 0,
                'correct_answer': 'Option A'
            }

# Example usage
if __name__ == "__main__":
    og = OptionGenerator()
    
    test_question = "What is the capital of France?"
    test_context = "Paris is the capital of France, known for its art, fashion, and culture."
    
    print(f"Question: {test_question}")
    print(f"Context: {test_context}")
    
    mcq = og.create_mcq_options(test_question, test_context)
    print("\nOptions:")
    for i, option in enumerate(mcq['options']):
        marker = "✓" if i == mcq['correct_index'] else " "
        print(f"{marker} {chr(65+i)}. {option}")