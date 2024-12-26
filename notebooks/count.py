import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "data/data.csv"
df = pd.read_csv(data_path)

print(df.columns)

# Display basic info and statistics
sns.countplot(data=df, x="diagnosis")
plt.title("Distribution of the target variable")
plt.show()
# Display basic info and statistics
sns.scatterplot(data=df, x="radius_mean", y="texture_mean", hue="diagnosis")
plt.title("Radius Mean and Texture Mean Distribution")
plt.show()


