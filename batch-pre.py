from transformers import BertTokenizer, BertForMaskedLM
import torch
import re

# Load the fine-tuned IndoBERT model and tokenizer
model_path = 'indolem/indobert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# Define a pattern to identify Roman numerals
roman_numeral_pattern = re.compile(r'^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$')

def is_roman_numeral(word):
    return bool(roman_numeral_pattern.match(word.upper()))

def highlight_and_mask_text(text):
    words = re.split(r'(\W+)', text)
    masked_text = []

    for word in words:
        if word.isalnum() and not word.isnumeric() and not is_roman_numeral(word):
            inputs = tokenizer(word, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_token_id = torch.argmax(logits[0, 1]).item()
            predicted_token = tokenizer.decode([predicted_token_id]).strip()

            if word.lower() != predicted_token.lower():
                masked_text.append('[MASK]')
            else:
                masked_text.append(word)
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

def process_articles(file_path, truth_file_path):
    with open(file_path, 'r') as file:
        original_texts = file.read().split('\n\n')  # Split articles by double newlines

    with open(truth_file_path, 'r') as file:
        ground_truth_texts = file.read().split('\n\n')

    detect_true = detect_false = correct_true = correct_false = final_true = final_false = total_words = 0
    article_count = len(original_texts)

    for idx, (original_text, ground_truth_text) in enumerate(zip(original_texts, ground_truth_texts), 1):
        # Sequentially process each article
        print(f"Article {idx}: read", end=", ")

        masked_text = highlight_and_mask_text(original_text)
        print("masked", end=", ")

        predicted_text = predict_masked_text(masked_text)
        print("predicted", end=", ")

        original_words = original_text.split()
        masked_words = masked_text.split()
        ground_truth_words = ground_truth_text.split()
        predicted_words = predicted_text.split()

        total_words += len(original_words)

        for orig_word, mask_word, pred_word, true_word in zip(original_words, masked_words, predicted_words, ground_truth_words):
            if not orig_word or not mask_word or not true_word or not pred_word:
                continue

            detect = (orig_word.lower() == true_word.lower() and mask_word == orig_word) or (orig_word.lower() != true_word.lower() and "[mask]" in mask_word.lower())
            correct = pred_word.lower() == true_word.lower()
            final = detect and correct

            if detect:
                detect_true += 1
            else:
                detect_false += 1

            if correct:
                correct_true += 1
            else:
                correct_false += 1

            if final:
                final_true += 1
            else:
                final_false += 1

        print("compared")

    detect_accuracy = detect_true / (detect_true + detect_false) if (detect_true + detect_false) > 0 else 0
    correct_accuracy = correct_true / (correct_true + correct_false) if (correct_true + correct_false) > 0 else 0
    overall_accuracy = final_true / (final_true + final_false) if (final_true + final_false) > 0 else 0

    print(f"\nBATCH PRETRAINED VERSION")
    # Final summary output
    print(f"\nArticle count: {article_count}")
    print(f"Total Word Count: {total_words}")

    print(f"\nDetection True: {detect_true}")
    print(f"Detection False: {detect_false}")
    print(f"Detection Accuracy: {detect_accuracy:.2f}")

    print(f"\nCorrection True: {correct_true}")
    print(f"Correction False: {correct_false}")
    print(f"Correction Accuracy: {correct_accuracy:.2f}")

    print(f"\nFinal True: {final_true}")
    print(f"Final False: {final_false}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

# Process the datasets
process_articles('dataset_error_sample.txt', 'dataset_truth_sample.txt')
