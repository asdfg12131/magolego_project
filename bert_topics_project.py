import pandas as pd
from bertopic import BERTopic

# Загружаем данные
df = pd.read_csv("reddit_parenting_with_comments.csv")

# Берём только тексты из столбца "body"
texts = df["body"].dropna().astype(str).tolist()

# Инициализируем модель
topic_model = BERTopic(language="english")

# Обучаем модель и получаем топики и вероятности
topics, probs = topic_model.fit_transform(texts)

# Создадим DataFrame с результатами
results_df = pd.DataFrame({
    "text": texts,
    "topic": topics,
    "probability": [prob.max() if prob is not None else None for prob in probs]
})

# Сохраним результаты с темами и вероятностями в CSV
results_df.to_csv("bertopic_results.csv", index=False)
print("Топики и вероятности сохранены в bertopic_results.csv")

# Получим частоты тем (без выбросов -1)
topic_freq = topic_model.get_topic_freq()
topic_freq = topic_freq[topic_freq.Topic != -1]

# Запишем топики и их ключевые слова в отдельный файл
with open("topics.txt", "w", encoding="utf-8") as f:
    for topic_id in topic_freq.Topic:
        topic_words = topic_model.get_topic(topic_id)
        # Берём только слова (без весов) и соединяем через запятую
        words_str = ", ".join([word for word, weight in topic_words])
        f.write(f"Topic {topic_id}: {words_str}\n")

print("Темы с ключевыми словами сохранены в topics.txt")

# Выведем несколько примеров из results_df
print(results_df.head(10))
