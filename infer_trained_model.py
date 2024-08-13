import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score

trained_model_dir = './phishing_detection_model'

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir, from_tf=False, config=trained_model_dir+'/config.json')
tokenizer = AutoTokenizer.from_pretrained('openai-community/openai-gpt')

def preprocess(urls):
    return tokenizer(urls, padding=False, truncation=True, return_tensors='pt')

def check_url_phishing(url):
    encodings = preprocess([url])
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # put the model in evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=-1).item()
    return "phishing" if prediction == 1 else "legitimate"

# Load the test dataset
data = pd.read_csv('./mnt/data/phishing_subset.csv')

# Optional limit in case it's too long
data = data.head(1000)

results = []
for index, row in data.iterrows():
    url = row['url']
    result = check_url_phishing(url)
    results.append({
        'url': url,
        'status': result
    })

results_df = pd.DataFrame(results)
results_df.to_csv('url_check_results.csv', index=False)

ground_truth_data = pd.read_csv('./mnt/data/phishing_subset.csv')

predictions_data = pd.read_csv('url_check_results.csv')

ground_truth_labels = ground_truth_data['status'].map({'phishing': 1, 'legitimate': 0})

predicted_labels = predictions_data['status'].map({'phishing': 1, 'legitimate': 0})

precision = precision_score(ground_truth_labels, predicted_labels)

print(f"Precision of the model: {precision:.4f}")

