import pandas as pd
pd.plotting.register_matplotlib_converters()
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
ign_filepath = "/home/zeddling/Documents/Personal Projects/Kaggle-Tutorials/Input/Visualization/ign_scores.csv"
ign_data = pd.read_csv(ign_filepath, index_col="Platform")

# Plot bar graph to determine if racing genre on a wii is a good choice for dev
plt.figure(figsize=(12, 5))
sns.barplot(x=ign_data.index, y=ign_data['Racing'])
plt.title("Average Score for Racing Games")
plt.ylabel("Average Score per Platform")

# Heatmap to determine best platform for development
plt.figure(figsize=(12, 6))
plt.title("Average Score Ratings")
sns.heatmap(data=ign_data, annot=True)
plt.xlabel("Genres")
plt.show()