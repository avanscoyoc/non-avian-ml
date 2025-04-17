import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# takes in folder path (folder of csv files to combine), output file that the single dataset will be in
# returns DataFrame
def merge_into_single_dataset(folder_path, output_file):
    file_lst = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    combined_df = pd.concat(
        (pd.read_csv(os.path.join(folder_path, file)) for file in file_lst),
          ignore_index=True)
    combined_df.to_csv(os.path.join(folder_path, output_file), index=False)
    return combined_df

def compute_mean_se(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(['model_name', 'species', 'training_size'])
    summary = grouped['avg_roc_auc'].agg(['mean', 'sem']).reset_index()
    summary.columns = ['model', 'species', 'train_size', 'mean_auc', 'std_err']
    return summary

def plot_roc_auc_by_train_size(df: pd.DataFrame, output_path: str) -> None:
    sns.set(style="whitegrid", font_scale=1.1)
    g = sns.FacetGrid(df, row="species",  sharey=True, height=4, aspect=1.5)
    models = df['model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    color_dict = dict(zip(models, palette))
    def plot_with_errorbars(data, **kwargs):
        for model, model_df in data.groupby("model"):
            plt.errorbar(
                model_df["train_size"],
                model_df["mean_auc"],
                yerr=model_df["std_err"],
                label=model,
                marker="o",
                capsize=4,
                linestyle="-",
                color = color_dict[model]
            )
        plt.legend(title="Model")

    g.map_dataframe(plot_with_errorbars)
    g.set_axis_labels("Training Size", "Mean ROC-AUC")
    g.set_titles(col_template="{col_name}")
    handles, labels = plt.gca().get_legend_handles_labels()
    g.fig.legend(handles, labels, title="Model", loc='upper right', ncol=len(models))
    g.fig.subplots_adjust(right=0.85, top = 0.85)
    #plt.tight_layout()
    plt.savefig(output_path, format='jpeg')
    plt.close()
    plt.close()
    print(f"Plot saved to {output_path}")

df = merge_into_single_dataset('/workspaces/non-avian-ml-toy/results', '/workspaces/tmp/merged_results.csv')
summary_df = compute_mean_se(df)
output_plot_path = "/workspaces/non-avian-ml-toy/tmp/roc_auc_plot.jpeg"
plot_roc_auc_by_train_size(summary_df, output_plot_path)