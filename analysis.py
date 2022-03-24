import os
import pathlib
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pickle
import plotly.graph_objects as go
import numpy as np
import collections
import json

pd.set_option("display.max_columns", None)
pd.set_option('mode.chained_assignment', None)


def data_parser(config_file: str) -> tuple[Any, Any, Any, Any, Any, Any, bool, bool, bool, bool, bool]:
    """
    Parse parameters from the json parameter file
    Parameters :
        config_file: path to a config file
    Return :
        List of parameters
    """
    parameters_json = json.load(open(config_file, "r"))
    df_filename = parameters_json["DataSetPath"]
    cols = list(parameters_json["FeaturesColumns"])
    seed = parameters_json["Seed"]
    pop_comparison = parameters_json["ComparisonColumn"]
    pcs_number = parameters_json["Number of PCs"]
    k_means_dict = parameters_json["Kmeans"]
    if isinstance(parameters_json["Show Violin"], bool):
        show_violin = parameters_json["Show Violin"]
    else:
        show_violin = bool(parameters_json["Show Violin"].lower() == "true")
    if isinstance(parameters_json["Save Violin"], bool):
        save_violin = parameters_json["Save Violin"]
    else:
        save_violin = bool(parameters_json["Save Violin"].lower() == "true")
    if isinstance(parameters_json["Save Model"], bool):
        save_model = parameters_json["Save Model"]
    else:
        save_model = bool(parameters_json["Save Model"].lower() == "true")
    if isinstance(parameters_json["Show cluster info"], bool):
        show_cluster_info = parameters_json["Show cluster info"]
    else:
        show_cluster_info = bool(parameters_json["Show cluster info"].lower() == "true")
    if isinstance(parameters_json["Save dataframe"], bool):
        save_df = parameters_json["Save dataframe"]
    else:
        save_df = bool(parameters_json["Save dataframe"].lower() == "true")
    return df_filename, cols, seed, pop_comparison, pcs_number, k_means_dict, show_violin, save_violin, \
        save_model, show_cluster_info, save_df


def general_analysis(config_file: str = "general_analysis_parameters.json"):
    """
    Analysis of the dataset based on a config file. One Standard Scaler and PCA will be trained. A K-means per population
    will be trained.
    Parameters :
        config_file: path to a config file
    """
    df_filename, cols, seed, pop_comparison, pcs_number, k_means_dict, show_violin, save_violin, \
    save_model, show_cluster_info, save_df = data_parser(config_file)

    if seed == "RANDOM":
        seed = np.random.randint(2147483647)
        print(seed)
    else:
        try:
            seed = int(seed)
        except ValueError:
            print("Seed should be RANDOM or an int.")

    pathlib.Path(f"{os.getcwd()}/saved_analysis/{seed}").mkdir(parents=True, exist_ok=True)

    pcs_number = int(pcs_number)
    if ".csv" in df_filename:
        df = pd.read_csv(df_filename)
    else:
        df = pd.read_excel(df_filename)
    df = df.dropna(subset=cols)
    # if df.compare(df2):
    #     print("NaN value had been found and removed")
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pcs_number, random_state=seed, svd_solver="full"),
    )
    print(df)
    print(cols)
    model.fit(df[cols])
    if save_model:
        pickle.dump(model,
                    open(f"saved_analysis/{seed}/PCA.pkl", "wb"))

    pcs = model.transform(df[cols])
    pca_cols = []
    subpop_dict = {}
    for pc in range(1, pcs_number + 1):
        df[f"Principal component {pc}"] = pcs[:, pc-1]
        pca_cols.append(f"Principal component {pc}")

    for subpop in np.unique(df[pop_comparison]):
        extracted_pop_df = df[df[pop_comparison] == subpop]
        subpop_kmeans = KMeans(n_clusters=int(k_means_dict[subpop]), random_state=seed).fit(extracted_pop_df[pca_cols])
        extracted_pop_df[f"{subpop} clusters"] = list(subpop_kmeans.labels_)
        subpop_dict[f"{subpop} clusters"] = extracted_pop_df

        if save_model:
            pickle.dump(subpop_kmeans,
                        open(f"saved_analysis/{seed}/k_means_{pop_comparison}_{subpop}_{k_means_dict[subpop]}_clusters.pkl", "wb"))
        if save_df:
            extracted_pop_df.to_csv(f"saved_analysis/{seed}/df_{pop_comparison}_{subpop}_{k_means_dict[subpop]}_{k_means_dict[subpop]}_clusters.csv")
        if show_cluster_info:
            extracted_pop_df[cols] = extracted_pop_df[cols].astype(np.float64)
            print("Mean : \n", extracted_pop_df[cols + pca_cols].mean())
            print("Mean of each cluster : \n", extracted_pop_df.groupby([f"{subpop} clusters"]).mean()[cols + pca_cols])
            print(f"{subpop} {k_means_dict[subpop]} clusters", collections.Counter(subpop_kmeans.labels_))
    if show_violin is True or save_violin is True:
        general_population_violin(subpop_dict, seed, cols, pca_cols, show_violin, save_violin)


def create_color_dict(subpopdf: dict[str, Any]) -> dict[str, Any]:
    """
    Determine a widespread range of color for x population and y cluster in each population based on HSV color scheme.
    h : corresponds to the number of populations
    s : corresponds to the number of clusters in this population

    Parameters :
     subpopdf: a list of Pandas DataFrame, each dataframe is an independent dataframe focus on one population

    Return :
     colorDict: a color dictionary with the key a combination of population and the cluster and the hsv color as value
    """
    colorDict = {}
    for i, (key, df) in enumerate(subpopdf.items()):
        for j, cluster in enumerate(np.unique(df[key])):
            colorDict[f"{key} {cluster}"] = \
                f"hsv({(i+1)*360/len(subpopdf.items())},100%,{(j+1)*100/len(np.unique(df[key]))}%)"
    return colorDict


def general_population_violin(subfeatures_df: dict[str, Any], seed: int, cols: list[str], pca_cols: list[str],
                              show_violin: bool, save_violin: bool):
    """
    Generate violin figure based on result of analysis
    Parameters :
     subfeatures_df: A list of Pandas DataFrame, each dataframe is an independent dataframe focus on one population
     seed : Seed of the analysis
     cols: A list of features columns to be plot
     pca_cols: The list of PCA columns to be plot
     show_violin: A boolean to show or not the final graph
     save_violin: A boolean to save or not the final graph

    Return :
     None
    """
    colorDict = create_color_dict(subfeatures_df)
    for feature in cols + pca_cols:
        fig = fig_layout(go.Figure())
        for key, df in subfeatures_df.items():
            for cluster in np.unique(df[key]):
                df2 = df[df[key] == cluster]
                violin_y = df2[feature]
                fig.add_trace(
                    go.Violin(y=violin_y,
                              x=[key] * len(df2),
                              name=f"{key} {cluster}", box={"visible": True}, points="all", meanline={"visible": True},
                              line_color=colorDict[f"{key} {cluster}"], opacity=0.6))
                fig.update_layout(title=feature)
        if show_violin:
            fig.show()
        if save_violin:
            fig.write_image(f"saved_analysis/{seed}/{feature}.svg")


def fig_layout(fig):
    """
    Custom white layout for plotly figure
    Parameters :
         fig: A plotly figure
    Return:
         A plotly figure with a specific theme applied
    """
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', ticks="outside", tickwidth=1, tickcolor='black',
                     ticklen=10)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks="outside", tickwidth=1, tickcolor='black',
                     ticklen=10)
    return fig


if __name__ == '__main__':
    general_analysis()
