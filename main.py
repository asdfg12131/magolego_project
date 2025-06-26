import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Загрузка данных
df = pd.read_csv("/Users/egor/Documents/magolego_project/reddit_parenting_qa_dataset.csv")

# 2. Предобработка текста
def clean_text(text):
    return str(text).replace("\n", " ").strip()

for col in ['question_title', 'question_body', 'answer']:
    df[col] = df[col].apply(lambda x: clean_text(x) if not pd.isna(x) else "")

df["full_question"] = df["question_title"] + " " + df["question_body"]

df = df.dropna(subset=['question_id', 'author'])
df['question_id'] = df['question_id'].astype(str)
df['author'] = df['author'].astype(str)

# 3. Извлечение тем с помощью BERTopic
vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.9)
topic_model = BERTopic(language="english", calculate_probabilities=False,
                       nr_topics="auto", vectorizer_model=vectorizer)
topics, probs = topic_model.fit_transform(df["full_question"].tolist())

df["topic"] = topics

# 4. Построение графа знаний
G = nx.Graph()

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Добавление узлов и рёбер"):
    question_id = str(row["question_id"])
    author = str(row["author"])
    topic = row["topic"]

    G.add_node(question_id, type="question", topic=topic)
    if author not in G:
        G.add_node(author, type="user", topic=-1)
    G.add_edge(question_id, author)

# 5. Node Embeddings с помощью Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, quiet=True)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

embedding_dict = {node: model.wv[node] for node in G.nodes()}

# 6. Преобразование графа в формат PyTorch Geometric
pyg_graph = from_networkx(G)

node_ids = list(G.nodes())
node_features = torch.tensor([embedding_dict[node] for node in node_ids], dtype=torch.float)

labels = []
for node in node_ids:
    if G.nodes[node]['type'] == 'question':
        try:
            labels.append(df[df["question_id"] == node]["topic"].values[0])
        except IndexError:
            labels.append(-1)
    else:
        labels.append(-1)

pyg_graph.x = node_features
pyg_graph.y = torch.tensor(labels, dtype=torch.long)

mask = pyg_graph.y != -1
train_mask = torch.zeros(len(mask), dtype=torch.bool)
test_mask = torch.zeros(len(mask), dtype=torch.bool)

train_idx = torch.randperm(int(mask.sum()))[:int(0.8 * mask.sum())]
train_mask[mask] = False
train_mask[mask.nonzero(as_tuple=True)[0][train_idx]] = True
test_mask = mask & ~train_mask

pyg_graph.train_mask = train_mask
pyg_graph.test_mask = test_mask

# 7. Определение и обучение GNN модели (GAT)
class GAT(nn.Module):
    def __init__(self, hidden_dim=64, heads=4, num_classes=len(set(topics))):
        super().__init__()
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gnn = GAT().to(device)
optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.005)
data_pyg = pyg_graph.to(device)

model_gnn.train()
for epoch in tqdm(range(100), desc="Обучение GAT"):
    optimizer.zero_grad()
    out = model_gnn(data_pyg)
    loss = F.nll_loss(out[data_pyg.train_mask], data_pyg.y[data_pyg.train_mask])
    loss.backward()
    optimizer.step()

    val_loss = F.nll_loss(out[data_pyg.test_mask], data_pyg.y[data_pyg.test_mask]).item()
    val_acc = (out.argmax(dim=1)[data_pyg.test_mask] == data_pyg.y[data_pyg.test_mask]).float().mean().item()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 8. Ранжирование ответов
df["rank_score"] = df["score"] + df["num_comments"]
top_answers = df.sort_values(by="rank_score", ascending=False).head(10)

# 9. Поиск ресурсов
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_resources(query, top_n=5):
    query_embedding = sentence_model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, topic_model.topic_embeddings_)
    top_topic_ids = np.argsort(similarity_scores[0])[-top_n:][::-1]

    return df[df["topic"].isin(top_topic_ids)].sort_values(by="rank_score", ascending=False).head(top_n)

query = "Как справиться со сном ребенка"
search_resources(query)[['question_title', 'topic', 'rank_score']]

# 10. Персонализация
def personalize(user_id, top_n=5):
    user_questions = df[df["author"] == user_id]["question_id"].unique()
    if len(user_questions) == 0:
        return df.groupby("question_id").first().sort_values(by="rank_score", ascending=False).head(top_n)
    user_topics = df[df["question_id"].isin(user_questions)]["topic"].mode()[0]
    recommendations = df[df["topic"] == user_topics].sort_values(by="rank_score", ascending=False)
    return recommendations.head(top_n)

sample_user = df.iloc[0]["author"]

# 11. Рекомендации
def recommend(user_id, query=None, top_n=5):
    if query:
        return search_resources(query, top_n=top_n)
    else:
        return personalize(user_id, top_n=top_n)

recommend(sample_user, query=query)[['question_title', 'topic', 'rank_score']]
