import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, GPT2Config
import torch
from torch.utils.data import Dataset

# Custom Dataset class
class URLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load the datasets
train_df = pd.read_csv('./mnt/data/phishing_train.csv')
validation_df = pd.read_csv('./mnt/data/phishing_validation.csv')
test_df = pd.read_csv('./mnt/data/phishing_subset.csv')

# Map 'status' to numerical labels
train_df['label'] = train_df['status'].map({'legitimate': 0, 'phishing': 1})
validation_df['label'] = validation_df['status'].map({'legitimate': 0, 'phishing': 1})
test_df['label'] = test_df['status'].map({'legitimate': 0, 'phishing': 1})

# Initialize the tokenizer
model_name = 'openai-community/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if not already present
eos_token = tokenizer.eos_token or '<|endoftext|>'
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': eos_token})
tokenizer.pad_token = eos_token

# Initialize the model configuration and set pad token id
configuration = GPT2Config.from_pretrained(model_name)
configuration.pad_token_id = configuration.eos_token_id
configuration.num_labels = 2  # Set number of labels here

# Load the model with the updated configuration
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=configuration)

# Resize token embeddings because we added a new special token
model.resize_token_embeddings(len(tokenizer))

# Tokenize the URLs
def tokenize_data(data):
    return tokenizer(data['url'].tolist(), padding=True, truncation=True, max_length=100, return_tensors='pt')

train_encodings = tokenize_data(train_df)
validation_encodings = tokenize_data(validation_df)
test_encodings = tokenize_data(test_df)

train_labels = train_df['label'].values
validation_labels = validation_df['label'].values
test_labels = test_df['label'].values

# Create the custom datasets
train_dataset = URLDataset(train_encodings, train_labels)
validation_dataset = URLDataset(validation_encodings, validation_labels)
test_dataset = URLDataset(test_encodings, test_labels)

# Prepare the training arguments with consistent evaluation and save strategies
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Set number of epochs
    per_device_train_batch_size=16,  # Batch size
    per_device_eval_batch_size=16,
    learning_rate=5e-5,  # Learning rate
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch',  # Align save strategy with evaluation strategy
    eval_strategy='epoch',  # Evaluate after each epoch
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=None  # Add your metric computation function here if needed
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Evaluate the model on the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test results:", test_results)

# Save the model
model.save_pretrained('./phishing_detection_model_gpt2')
topenai-community/gpt2okenizer.save_pretrained('./phishing_detection_model_gpt2')

