from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForMaskedLM
import torch
import threading
import re
import spacy

# Initialize Flask app with specified folders
app = Flask(__name__, template_folder='./views', static_folder='./static')

# Load fine-tuned IndoBERT model and tokenizer at startup
print("Loading fine-tuned IndoBERT model...")
model_path = 'indolem/indobert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)
print("Model loaded successfully.")

# Create a lock object for thread-safe model access
model_lock = threading.Lock()

# Load the KBBI words
with open('./static/full_kbbi.txt', 'r', encoding='utf-8') as file:
    kbbi_words = set(word.strip() for word in file.readlines())

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Affix handling
prefixes = ["me", "di", "ke", "se", "te", "ber", "ter", "per", "non", "meng"]
suffixes = ["an", "kan", "i", "nya"]

# Roman numeral pattern
roman_numeral_pattern = re.compile(r'^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$')

def strip_affixes(word):
    # Strip known prefixes and suffixes
    for prefix in prefixes:
        if word.startswith(prefix):
            stripped_word = word[len(prefix):]
            if stripped_word in kbbi_words:
                return stripped_word
            for suffix in suffixes:
                if stripped_word.endswith(suffix):
                    stripped_word2 = stripped_word[:-len(suffix)]
                    if stripped_word2 in kbbi_words:
                        return stripped_word2
    # Check if stripping suffixes first works
    for suffix in suffixes:
        if word.endswith(suffix):
            stripped_word = word[:-len(suffix)]
            if stripped_word in kbbi_words:
                return stripped_word
            for prefix in prefixes:
                if stripped_word.startswith(prefix):
                    stripped_word2 = stripped_word[len(prefix):]
                    if stripped_word2 in kbbi_words:
                        return stripped_word2
    return word

def is_name(word):
    doc = nlp(word)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return True
    return False

def highlight_and_mask_text(text):
    words = re.split(r'(\W+)', text)
    highlighted_text = []
    masked_text = []
    for word in words:
        stripped_word = strip_affixes(word.lower())
        if (stripped_word not in kbbi_words and not word.isnumeric() and not roman_numeral_pattern.match(word.upper()) and not is_name(word) and word.isalnum()):
            highlighted_text.append(f'<span class="spanred">{word}</span>')
            masked_text.append('[MASK]')
        else:
            highlighted_text.append(word)
            masked_text.append(word)
    return ''.join(highlighted_text), ''.join(masked_text)

def predict_masked_text(masked_text):
    inputs = tokenizer(masked_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    masked_indices = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    predicted_tokens = []
    for idx in masked_indices:
        logits_idx = logits[0, idx]
        predicted_token_id = torch.argmax(logits_idx).item()
        predicted_token = tokenizer.decode([predicted_token_id])
        predicted_tokens.append(predicted_token)
    corrected_text = masked_text
    for token in predicted_tokens:
        corrected_text = corrected_text.replace('[MASK]', f'<span class="spangreen">{token}</span>', 1)
    return corrected_text

def split_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        word_length = len(tokenizer.tokenize(word))
        if current_length + word_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/full_kbbi.txt')
def kbbi():
    return app.send_static_file('full_kbbi.txt')

@app.route('/text-correction', methods=['POST'])
def text_correction():
    data = request.form.get('data', '')
    chunks = split_text(data)
    highlighted_chunks = []
    masked_chunks = []
    corrected_chunks = []

    for chunk in chunks:
        highlighted_text, masked_text = highlight_and_mask_text(chunk)
        corrected_text = predict_masked_text(masked_text)
        highlighted_chunks.append(highlighted_text)
        masked_chunks.append(masked_text)
        corrected_chunks.append(corrected_text)

    response = {
        'highlighted': ' '.join(highlighted_chunks),
        'masked': ' '.join(masked_chunks),
        'corrected': ' '.join(corrected_chunks)
    }
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

# Main block to ensure the model is loaded before starting the server
if __name__ == '__main__':
    try:
        print("Ensuring model is loaded...")
        test_input = "test [MASK]"
        predict_masked_text(test_input)
        print("Model is ready, starting the server...")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    app.run(port=5000, debug=True)
