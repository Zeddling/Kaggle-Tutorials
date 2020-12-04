import pandas as pd
pd.plotting.register_matplotlib_converters()
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
museum_filepath = "/home/zeddling/Documents/Personal Projects/Kaggle-Tutorials/Input/Visualization/museum_visitors.csv"
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)

# Review Data
# print(museum_data.tail())

# Plot graph showing visitors traffic against time
plt.figure(figsize=(7, 6))
sns.lineplot(data=museum_data)
plt.title("Monthly Visitors to Los Angeles City Museums")
# plt.show()

# Plot Avila Adobe graph
plt.figure(figsize=(5,5))
sns.lineplot(data=museum_data['Avila Adobe'])
plt.title("Monthly Visitors to Avila Adobe")
plt.xlabel("Date")
plt.show()
