from datasets import load_dataset

# Load the dataset from Hugging Face Hub
dataset = load_dataset("mlfoundations-dev/sandboxes-tasks", split='train')

# Print basic info about the dataset
print(dataset)
