import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_hyperplane(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    print(f"Data loaded from: {file_path}\n")
    print(f"Dataframe shape: {df.shape}\n")

    class_counts = df['class'].value_counts()
    print("Hyperplane: Class Distribution\n", class_counts)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='class')
    plt.title("Hyperplane: Class Distribution")
    plt.tight_layout()
    
    bar_plot_path = os.path.join(output_dir, "Hyperplane_class_distribution.png")
    plt.savefig(bar_plot_path, dpi=120)
    plt.show()
    print(f"Class distribution plot saved to: {bar_plot_path}\n")

    numeric_cols = [col for col in df.columns if col.startswith("att")]
    grouped_stats = df.groupby('class')[numeric_cols].describe()
    print("Descriptive Statistics per Class for att1–att10:\n", grouped_stats, "\n")

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
    fig.suptitle("Hyperplane: Boxplots of Numeric Attributes by Class", fontsize=16)

    for i, col in enumerate(numeric_cols):
        r, c = divmod(i, 5)
        sns.boxplot(data=df, x='class', y=col, ax=axes[r][c])
        axes[r][c].set_title(f"{col} by Class")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    boxplot_path = os.path.join(output_dir, "Hyperplane_attribute_boxplots.png")
    plt.savefig(boxplot_path, dpi=120)
    plt.show()
    print(f"Attribute boxplots saved to: {boxplot_path}\n")


if __name__ == "__main__":
    input_file = "/Users/jackfurmston/developer/cybersecurity_game/data/Concept Drift and Adaptive Learning/OpenML/HYPERPLANE_01_converted.csv"
    output_path = "/Users/jackfurmston/developer/cybersecurity_game/docs/Data Analytics Results"

    analyze_hyperplane(input_file, output_path)




