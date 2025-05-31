#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libaries
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter


# In[3]:


df = pd.read_csv("gpt_dataset.csv")


df.dropna(subset=["Category","Resume"],inplace=True)

print(df.head())
print(df.isna().sum())


# In[4]:


nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


df["processed_resume"] = df["Resume"].apply(preprocess_text)


# In[5]:


# Function to extract keywords from text using SpaCy
def extract_keywords(text, top_n=10):
    """
    Extract top_n keywords from the text using SpaCy.
    Keywords are nouns, proper nouns, and adjectives.
    """
    doc = nlp(text)
    keywords = [
        token.text.lower() for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN", "ADJ"]
    ]
    keyword_counts = Counter(keywords)
    return [word for word, _ in keyword_counts.most_common(top_n)]


# In[6]:


def extract_category_keywords(df, column="resume", category_column="category", top_n=10):
    """
    Extract top_n keywords for each category in the dataset.
    """
    category_keywords = defaultdict(list)

    # Group resumes by category
    grouped = df.groupby(category_column)[column].apply(" ".join)

    for category, resumes in grouped.items():
        keywords = extract_keywords(resumes, top_n=top_n)
        category_keywords[category] = keywords

    return category_keywords


# In[7]:


category_keywords = extract_category_keywords(df, column="Resume", category_column="Category", top_n=10)


# In[8]:


label_encoder = LabelEncoder()

# Encode the 'category' column
df["category_encoded"] = label_encoder.fit_transform(df["Category"])

# Inspect the encoded labels
print(df[["Category", "category_encoded"]].head())


# In[9]:


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for simplicity

# Fit and transform the processed text data
X = vectorizer.fit_transform(df["processed_resume"])

# Convert to a dense array (if needed)
X = X.toarray()

# Inspect the shape of the feature matrix
print(X.shape)  # (num_samples, num_features)


# In[10]:


# Extract encoded labels
y = df["category_encoded"].values

# Convert labels to PyTorch tensor
y_tensor = torch.tensor(y, dtype=torch.long)  # Use torch.long for classification


# In[11]:


# Convert features to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# Inspect the tensors
print(X_tensor.shape)  # (num_samples, num_features)
print(y_tensor.shape)  # (num_samples,)


# In[12]:


from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Inspect the shapes
print(X_train.shape, X_test.shape)  # Training and testing feature sets
print(y_train.shape, y_test.shape)  # Training and testing labels


# In[13]:


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
class ResumeClassifier(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(ResumeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Hidden layer to output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



# In[14]:


input_size = X_train.shape[1]  # Number of features (5000 from TF-IDF)
hidden_size = 128              # Size of the hidden layer
num_classes = len(df["category_encoded"].unique())  # Number of unique categories
learning_rate = 0.001
num_epochs = 10
batch_size = 32


# In[15]:


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# In[16]:


model = ResumeClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[17]:


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# In[18]:


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# In[19]:


def score_resume(resume_text, model, vectorizer, label_encoder):
    # Preprocess the resume text
    processed_text = preprocess_text(resume_text)

    # Vectorize the resume text
    resume_vector = vectorizer.transform([processed_text]).toarray()
    resume_tensor = torch.tensor(resume_vector, dtype=torch.float32)

    # Predict the category
    model.eval()
    with torch.no_grad():
        outputs = model(resume_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Decode the predicted category
    predicted_category = label_encoder.inverse_transform(predicted_class.numpy())[0]
    return predicted_category, confidence.item()


# In[20]:


def calculate_resume_quality(resume_text, predicted_category, category_keywords):
    """
    Calculate a quality score for the resume based on multiple criteria.
    """
    # Preprocess the resume text
    processed_text = preprocess_text(resume_text)

    # Keyword coverage: Check for relevant keywords in the resume
    keywords = category_keywords.get(predicted_category, [])
    keyword_coverage = sum(1 for keyword in keywords if keyword in processed_text.lower()) / len(keywords) if keywords else 0

    # Completeness: Check for essential sections (e.g., "experience", "education", "skills")
    essential_sections = ["experience", "education", "skills"]
    completeness_score = sum(1 for section in essential_sections if section in processed_text.lower()) / len(essential_sections)

    # Combine scores into a final quality score
    quality_score = (
        0.6 * keyword_coverage +       # Weight for keyword coverage
        0.4 * completeness_score       # Weight for completeness
    )

    return quality_score, keyword_coverage, completeness_score


# In[21]:


from docx import Document  # Library to extract text from DOCX files

def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

def score_resume_from_docx(file_path, model, vectorizer, label_encoder, category_keywords):
    # Extract text from the .docx file
    resume_text = extract_text_from_docx(file_path)

    # Preprocess the resume text
    processed_text = preprocess_text(resume_text)

    # Vectorize the resume text
    resume_vector = vectorizer.transform([processed_text]).toarray()
    resume_tensor = torch.tensor(resume_vector, dtype=torch.float32)

    # Predict the category
    model.eval()
    with torch.no_grad():
        outputs = model(resume_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Decode the predicted category
    predicted_category = label_encoder.inverse_transform(predicted_class.numpy())[0]

    # Calculate the quality score
    quality_score, keyword_coverage, completeness_score = calculate_resume_quality(
        resume_text, predicted_category, category_keywords
    )

    return {
        "predicted_category": predicted_category,
        "confidence_score": confidence.item(),
        "quality_score": quality_score,
        "keyword_coverage": keyword_coverage,
        "completeness_score": completeness_score
    } 


# In[ ]:


docx_resume_path = "CURRICULUMVITAE.docx"  # Replace with the path to your DOCX resume
# Score the resume
# Score the resume
results = score_resume_from_docx(docx_resume_path, model = model, vectorizer = vectorizer, label_encoder = label_encoder, category_keywords = category_keywords)

# Print the results
print(f"Predicted Category: {results['predicted_category']}")
print(f"Confidence Score: {results['confidence_score']:.4f}")
print(f"Quality Score: {results['quality_score']:.4f}")
print(f"Keyword Coverage: {results['keyword_coverage']:.4f}")
print(f"Completeness Score: {results['completeness_score']:.4f}")

