import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup


class DataProcessor:
    def __init__(self, file_path, stop_words=[]):
        super(DataProcessor, self).__init__()
        self.fp = file_path
        self.stop_words = set(stop_words)

    def load_data(self):
        df = pd.read_csv(self.fp, encoding='utf-8')
        return df

    def process_data(self, df):
        df['reviewText'] = df['reviewText'].fillna('Missing')
        df['reviews'] = df['summary'] + '. ' + df['reviewText']

        def func(row):
            if row['overall'] >= 4.0:
                return 'positive'
            else:
                return 'negative'

        df['sentiment'] = df.apply(func, axis=1)
        df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
        return df

    def vectorize(self, df):
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
        X = vectorizer.fit_transform(df['reviews'])
        y = df['sentiment']
        return X, y
    
    def over_sampling(self, X, y):
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        return X_smote, y_smote
    

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

"""
机器学习方法
"""
class MLModel:
    def __init__(self):
        self.lr = LogisticRegression(random_state=42)
        self.dt = DecisionTreeClassifier()
        self.knn = KNeighborsClassifier()
        self.svc = SVC()
        self.bnb = BernoulliNB()

    def train_models(self, X_train, y_train):
        self.lr.fit(X_train, y_train)
        # self.dt.fit(X_train, y_train)
        # self.knn.fit(X_train, y_train)
        self.svc.fit(X_train, y_train)
        # self.bnb.fit(X_train, y_train)

    def evaluate_models(self, X_test, y_test):
        print('Logistic Regression:', cross_val_score(self.lr, X_test, y_test, cv=5, scoring='accuracy').mean())
        # print('Decision Tree:', cross_val_score(self.dt, X_test, y_test, cv=5, scoring='accuracy').mean())
        # print('KNN:', cross_val_score(self.knn, X_test, y_test, cv=5, scoring='accuracy').mean())
        print('SVC:', cross_val_score(self.svc, X_test, y_test, cv=5, scoring='accuracy').mean())
        # print('BernoulliNB:', cross_val_score(self.bnb, X_test, y_test, cv=5, scoring='accuracy').mean())

    def grid_search_lr(self, X_train, y_train):
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l2']
            }
        grid_search = GridSearchCV(self.lr, param_grid, cv=5, verbose=0, n_jobs=-1)
        best_model = grid_search.fit(X_train, y_train)
        print('Best parameters of Logistic Regression:', best_model.best_params_)
        print('Best score of Logistic Regression:', best_model.best_score_)
        return best_model.best_params_
    

"""
深度学习方法
"""

EPOCHS = 3
MAX_LEN = 128
BATCH_SIZE = 16
MODEL_NAME = "google-bert/bert-base-uncased"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DLModel(nn.Module):
    def __init__(self, model, n_classes):
        super(DLModel, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return output
    
    def set_device(self):
        self.to(device)
        print("Model on device: ", device)


class DLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        super(DLDataset, self).__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    losses = []
    corr_preds = 0

    for batch in train_loader:
        inputs_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        corr_preds += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return np.mean(losses), corr_preds.double() / len(train_loader.dataset)

def eval_model(model, val_loader, criterion, device):
    model.eval()
    losses = []
    corr_preds = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            corr_preds += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = corr_preds.double() / len(val_loader.dataset)
    report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])
    return np.mean(losses), acc, report


if __name__ == '__main__':

    print("Machine Learning Model: ")
    print()
    processer = DataProcessor('./Musical_instruments_reviews.csv')
    df = processer.load_data()
    df = processer.process_data(df)
    X, y = processer.vectorize(df)
    X, y = processer.over_sampling(X, y)
    X_train, X_test, y_train, y_test = processer.split_data(X, y)

    model = MLModel()
    model.train_models(X_train, y_train)
    model.evaluate_models(X_test, y_test)
    model.grid_search_lr(X_train, y_train)
    print()
    
    # --------------------------------------------------------------------

    print("Deep Learning Model: ")
    print()
    processer = DataProcessor('./Musical_instruments_reviews.csv')
    df = processer.load_data()
    df = processer.process_data(df)
    X_train, X_test, y_train, y_test = processer.split_data(df['reviews'].values, df['sentiment'].values)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertModel.from_pretrained(MODEL_NAME)

    model = DLModel(bert_model, 2)
    model.set_device()

    train_dataset = DLDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = DLDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        start_time = time.time()
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_report = eval_model(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.4f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.4f}%')
        if epoch == EPOCHS-1:
            print(val_report)


"""
Machine Learning Model: 

Logistic Regression: 0.8564662806735797
SVC: 0.9132702732815687
Best parameters of Logistic Regression: {'C': 1438.44988828766, 'penalty': 'l2'}
Best score of Logistic Regression: 0.9493591963976445



Deep Learning Model: 

Model on device:  cuda
Epoch 1/3
----------
Time: 2.0m 15.74904489517212s
Train Loss: 0.2381 | Train Acc: 90.9844%
Val Loss: 0.1861 | Val Acc: 93.9113%
Epoch 2/3
----------
Time: 2.0m 11.075915098190308s
Train Loss: 0.1414 | Train Acc: 95.5044%
Val Loss: 0.2207 | Val Acc: 94.1062%
Epoch 3/3
----------
Time: 2.0m 11.26940631866455s
Train Loss: 0.0786 | Train Acc: 98.0019%
Val Loss: 0.2583 | Val Acc: 93.7652%

              precision    recall  f1-score   support

    negative       0.75      0.70      0.73       244
    positive       0.96      0.97      0.96      1809

    accuracy                           0.94      2053
   macro avg       0.86      0.84      0.85      2053
weighted avg       0.94      0.94      0.94      2053
"""