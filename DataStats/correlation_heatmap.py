import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filePath = r"C:\Users\sol77\Desktop\ML-For-CV-Robustness\datasets\trees-dl-real.csv"
df = pd.read_csv(filePath)

numeric_df = df.select_dtypes(include="number")

corr_matrix = numeric_df.corr(method="pearson")

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,          
    fmt=".2f",           
    cmap="coolwarm",     
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson correlation coefficient"}
)

plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(r"C:\Users\sol77\Desktop\correlation_trees-dl-real.png", dpi=300, bbox_inches="tight")
plt.show()