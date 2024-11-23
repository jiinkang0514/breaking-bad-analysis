# Install necessary libraries (uncomment this line if running in Colab or locally without libraries installed)
# !pip install nltk requests beautifulsoup4 pandas matplotlib seaborn wordcloud networkx

# Import libraries
import nltk
import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
import re
import numpy as np
import itertools
from collections import defaultdict
import networkx as nx
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Base URL for relative links
base_url = "https://transcripts.foreverdreaming.org"

# URL of the forum page with all episodes
forum_url = "https://transcripts.foreverdreaming.org/viewforum.php?f=165"

# Fetch episode links from the forum page
response = requests.get(forum_url)
soup = BeautifulSoup(response.text, "html.parser")
episode_links = soup.select(".row-item.topic_read .list-inner a.topictitle")
print(f"Found {len(episode_links)} episode links.")

# Initialize a list to store data
data = []

# Loop through each episode link and scrape data
for link in episode_links:
    try:
        episode_url = base_url + link["href"][1:]
        episode_title = link.text.strip()

        # Request the episode transcript page
        response = requests.get(episode_url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the transcript content
        transcript = soup.select_one(".content").get_text(separator="\n")

        # Append data to the list
        data.append([episode_title, episode_url, transcript])
    except Exception as e:
        print(f"Error processing {link}: {e}")
        continue

# Convert the scraped data into a DataFrame
df_raw = pd.DataFrame(data, columns=["Episode Title", "URL", "Transcript"])

# Initialize a set to store unique character names
unique_chars = set()
for _, row in df_raw.iterrows():
    lines = row["Transcript"].split("\n")
    for line in lines:
        if ":" in line:
            char_name = line.split(":")[0].strip()
            if char_name and " " not in char_name:
                unique_chars.add(char_name)

# Add main characters to the set
main_chars = {
    "Walter", "Walt", "Skyler", "Jesse", "Hank", "Steve", "Marie", "Saul",
    "Gus", "Mike", "Todd", "Tim", "Tuco", "Mrs. Pinkman", "Walter Jr", "Dr. Delcavoli"
}
all_chars = unique_chars.union(main_chars)

# Process dialogue data
data = []
for _, row in df_raw.iterrows():
    episode_title = row["Episode Title"]
    episode_url = row["URL"]
    transcript = row["Transcript"]

    lines = transcript.split("\n")
    for line in lines:
        for char in all_chars:
            if char + ":" in line:
                character, dialogue = line.split(":", 1)
                data.append([episode_title, episode_url, char.strip(), dialogue.strip()])

# Create a DataFrame with processed data
df = pd.DataFrame(data, columns=["Episode Title", "URL", "Character", "Dialogue"])

# Ensure consistent character names
df["Character"] = df["Character"].apply(lambda x: "Walter" if x == "Walt" else x)

# Map episode titles to episode numbers
episode_map = {
    f"{season}x{episode:02}": (season - 1) * 7 + episode - 1
    for season in range(1, 4) for episode in range(1, 8)
}
df["Episode Number"] = df["Episode Title"].apply(lambda title: episode_map.get(title[:4], -1))

# Add sentiment analysis using TextBlob
df["Sentiment"] = df["Dialogue"].apply(lambda dialogue: TextBlob(dialogue).sentiment.polarity)

# Define the set of valid characters
valid_characters = {"Walter", "Jesse", "Skyler", "Scene", "Hank", "Marie", "Walter Jr"}

# Filter the DataFrame to include only valid characters
df_filtered = df[df["Character"].isin(valid_characters)]

# Calculate average sentiment polarity for each valid character
sentiment_by_character = df_filtered.groupby("Character")["Sentiment"].mean().sort_values(ascending=False)

# Visualize the average sentiment by character
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_by_character.index, y=sentiment_by_character.values, palette="viridis")
plt.xlabel("Character")
plt.ylabel("Average Sentiment Polarity")
plt.title("Average Sentiment by Character (Filtered)")
plt.xticks(rotation=45)
plt.show()

# Visualization: Sentiment Over Episodes
avg_sentiment_per_episode = df.groupby("Episode Number")["Sentiment"].mean()
plt.figure(figsize=(14, 7))
plt.plot(avg_sentiment_per_episode.index, avg_sentiment_per_episode.values, marker='o')
plt.xlabel("Episode Number")
plt.ylabel("Average Sentiment Polarity")
plt.title("Average Sentiment Over Episodes")
plt.show()

# Correlation Matrix for Top Characters
# Group by episode number and character, count number of lines per character per episode
df_lines = df.groupby(["Episode Number", "Character"])["Dialogue"].count().reset_index()

# Pivot the DataFrame to create a character vs. episode matrix
df_pivot = df_lines.pivot(index="Episode Number", columns="Character", values="Dialogue").fillna(0)

# Filter to include only the top characters
top_characters = df["Character"].value_counts().head(7).index.tolist()
filtered_pivot = df_pivot[top_characters]

# Calculate the correlation matrix for top characters
corr_matrix = filtered_pivot.corr()

# Plot the filtered correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.5})
plt.title("Correlation of Importance Between Top Characters")
plt.show()

# Character Interaction Graph
interaction_counts = defaultdict(int)

# Count mentions between top characters
for episode, group in df.groupby("Episode Number"):
    episode_dialogues = group[group["Character"].isin(top_characters)]
    for char1, char2 in itertools.combinations(top_characters, 2):
        if char1 in " ".join(episode_dialogues[episode_dialogues["Character"] == char2]["Dialogue"]) or \
           char2 in " ".join(episode_dialogues[episode_dialogues["Character"] == char1]["Dialogue"]):
            interaction_counts[(char1, char2)] += 1

# Create a graph
interaction_graph = nx.Graph()
for char in top_characters:
    interaction_graph.add_node(char)
for (char1, char2), weight in interaction_counts.items():
    interaction_graph.add_edge(char1, char2, weight=weight)

# Draw the interaction graph
plt.figure(figsize=(12, 8))
weights = [interaction_graph[u][v]["weight"] for u, v in interaction_graph.edges()]
nx.draw_circular(interaction_graph, 
                 with_labels=True, 
                 font_size=12, 
                 node_size=3000, 
                 node_color="skyblue", 
                 edge_color="black", 
                 width=weights)
plt.title("Mentions Between Main Characters")
plt.show()

# Word Cloud for All Dialogues
stop_words = set(stopwords.words("english")).union({"yeah", "right", "okay"})
wordcloud = WordCloud(width=1280, height=720, max_words=150, stopwords=stop_words).generate(" ".join(df["Dialogue"]))
plt.figure(figsize=(14, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of All Dialogues")
plt.show()

# Word Clouds for Top Characters
for character in top_characters:
    dialogues = " ".join(df[df["Character"] == character]["Dialogue"])
    wordcloud = WordCloud(width=1920, height=1080, max_words=150, stopwords=stop_words).generate(dialogues)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {character}")
    plt.show()
  
