import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score

def preprocess(urls, tokenizer):
    return tokenizer(urls, padding=False, truncation=True, return_tensors='pt')

def check_url_phishing(url, model, tokenizer):
    encodings = preprocess([url], tokenizer)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # put the model in evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=-1).item()
    return "phishing" if prediction == 1 else "legitimate"

def evaluate_model(model_name, data):
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []
    for index, row in data.iterrows():
        url = row['url']
        result = check_url_phishing(url, model, tokenizer)
        results.append({
            'url': url,
            'status': result
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Load the test dataset
data = pd.read_csv('./mnt/data/phishing_subset.csv')

# Optional limit in case it's too long
data = data.head(1000)

custom_model_name = './phishing_detection_model_gpt2'
custom_model_results_df = evaluate_model(custom_model_name, data)

gpt_model_name = 'openai-community/gpt2'
gpt_model_results_df = evaluate_model(gpt_model_name, data)

ground_truth_labels = data['status'].map({'phishing': 1, 'legitimate': 0})

custom_predicted_labels = custom_model_results_df['status'].map({'phishing': 1, 'legitimate': 0})
gpt_predicted_labels = gpt_model_results_df['status'].map({'phishing': 1, 'legitimate': 0})

custom_model_precision = precision_score(ground_truth_labels, custom_predicted_labels)
gpt_model_precision = precision_score(ground_truth_labels, gpt_predicted_labels)

print(f"Precision of the custom-trained model: {custom_model_precision:.4f}")
print(f"Precision of the openai-community/gpt2 model: {gpt_model_precision:.4f}")

