import matplotlib.pyplot as plt
import pandas as pd

# Məlumatları oxuyun
data = pd.read_csv('data.csv')

# Target dəyişəninin paylanması
data['id'].value_counts().plot(kind='bar')
plt.title("Target Distribution")
plt.savefig("target_distribution.png")  # Qrafiki fayla saxlayır
plt.close()
