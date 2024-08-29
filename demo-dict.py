from transformers import BertTokenizer, BertForMaskedLM
import torch
import re
import spacy

# Load the fine-tuned IndoBERT model and tokenizer
model_path = 'indolem/indobert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

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
    masked_text = []
    for word in words:
        stripped_word = strip_affixes(word.lower())
        if (stripped_word not in kbbi_words and not word.isnumeric() and not roman_numeral_pattern.match(word.upper()) and not is_name(word) and word.isalnum()):
            masked_text.append('[MASK]')
        else:
            masked_text.append(word)
    return ''.join(masked_text)

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
        corrected_text = corrected_text.replace('[MASK]', token, 1)
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

def process_text_file(file_path, truth_file_path):
    with open(file_path, 'r') as file:
        original_text = file.read().strip()

    with open(truth_file_path, 'r') as file:
        ground_truth_text = file.read().strip()

    # Generate masked text and predicted text
    chunks = split_text(original_text)
    masked_chunks = []
    predicted_chunks = []

    for chunk in chunks:
        masked_text = highlight_and_mask_text(chunk)
        predicted_text = predict_masked_text(masked_text)
        masked_chunks.append(masked_text)
        predicted_chunks.append(predicted_text)

    masked_text = ' '.join(masked_chunks)
    predicted_text = ' '.join(predicted_chunks)

    original_words = original_text.split()
    masked_words = masked_text.split()
    ground_truth_words = ground_truth_text.split()
    predicted_words = predicted_text.split()

    final_true = 0
    final_false = 0
    total_detect_correct = 0
    total_detect_incorrect = 0
    total_correct_correct = 0
    total_correct_incorrect = 0

    classifications = []

    # Add the input, masked, predicted, and ground truth sections
    print("\nInput:")
    print(original_text)
    print("\nMasked:")
    print(masked_text)
    print("\nPredicted:")
    print(predicted_text)
    print("\nGround Truth:")
    print(ground_truth_text)

    # Table headers for the new format
    print("\nOriginal Table with True/False Classification:\n")
    print(f"|{'-'*7}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|")
    print(f"| {'Index':<5} | {'Input':<20} | {'Masked':<20} | {'Predict':<20} | {'Truth':<20} | {'Detect':<8} | {'Correct':<8} | {'Final':<8} |")
    print(f"|{'-'*7}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|")

    for i, (orig_word, mask_word, pred_word, true_word) in enumerate(zip(original_words, masked_words, predicted_words, ground_truth_words), 1):
        if not orig_word or not mask_word or not true_word or not pred_word:
            continue

        # Check detection correctness
        detect = (orig_word.lower() == true_word.lower() and mask_word == orig_word) or (orig_word.lower() != true_word.lower() and "[mask]" in mask_word.lower())

        # Check correction correctness
        correct = pred_word.lower() == true_word.lower()

        # Final correctness (True if both detect and correct are True)
        final = detect and correct

        # Update counters
        if final:
            final_true += 1
        else:
            final_false += 1

        if detect:
            total_detect_correct += 1
        else:
            total_detect_incorrect += 1

        if correct:
            total_correct_correct += 1
        else:
            total_correct_incorrect += 1

        # Store the classification results
        classifications.append((i, orig_word, mask_word, pred_word, true_word, detect, correct, final))

        # Print the updated table row
        print(f"| {i:<5} | {orig_word:<20} | {mask_word:<20} | {pred_word:<20} | {true_word:<20} | {str(detect):<8} | {str(correct):<8} | {str(final):<8} |")
    print(f"|{'-'*7}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|")

    # Sort classifications based on the 'Final' column (True first, then False) while maintaining original index order
    classifications_sorted = sorted(classifications, key=lambda x: (not x[7], x[0]))

    # Table headers for the sorted output
    print("\nSorted Table Based on Final (True, then False):\n")
    print(f"|{'-'*7}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|")
    print(f"| {'Index':<5} | {'Input':<20} | {'Masked':<20} | {'Predict':<20} | {'Truth':<20} | {'Detect':<8} | {'Correct':<8} | {'Final':<8} |")
    print(f"|{'-'*7}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|")

    for i, orig_word, mask_word, pred_word, true_word, detect, correct, final in classifications_sorted:
        # Print the sorted table row
        print(f"| {i:<5} | {orig_word:<20} | {mask_word:<20} | {pred_word:<20} | {true_word:<20} | {str(detect):<8} | {str(correct):<8} | {str(final):<8} |")
    print(f"|{'-'*7}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|")

    # Calculate and print evaluation metrics
    accuracy = final_true / (final_true + final_false) if (final_true + final_false) > 0 else 0
    detect_accuracy = total_detect_correct / (total_detect_correct + total_detect_incorrect) if (total_detect_correct + total_detect_incorrect) > 0 else 0
    correction_accuracy = total_correct_correct / (total_correct_correct + total_correct_incorrect) if (total_correct_correct + total_correct_incorrect) > 0 else 0

    print(f"\nDEMO DICTIONARY VERSION")

    word_count = len(original_words)

    print(f"\nWord Count: {word_count}")

    print(f"\nDetection True: {total_detect_correct}")
    print(f"Detection False: {total_detect_incorrect}")
    print(f"Detection Accuracy: {detect_accuracy:.2f}")

    print(f"\nCorrection True: {total_correct_correct}")
    print(f"Correction False: {total_correct_incorrect}")
    print(f"Correction Accuracy: {correction_accuracy:.2f}")

    print(f"\nFinal True: {final_true}")
    print(f"Final False: {final_false}")
    print(f"Overall Accuracy: {accuracy:.2f}")

# Test with the provided dummy_text and dummy_truth files
process_text_file('dummy_text.txt', 'dummy_truth.txt')
