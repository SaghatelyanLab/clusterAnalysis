import os
import pathlib
from typing import Tuple, Any

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
import re
from analysis import general_population_violin


def data_parser(config_file: str) -> tuple[Any,  Any, Any, Any, Any, bool, bool, bool, bool, bool]:
    """
    Parse parameters from the json parameter file
    Parameters:
        config_file: path to a config file
    Return:
         List of parameters
    """
    parameters_json = json.load(open(config_file, "r"))

    df_filepath = parameters_json["DataSetPath"]
    cols = parameters_json["FeaturesColumns"]
    pop_comparison = parameters_json["ComparisonColumn"]
    pca_filepath = parameters_json["PCA model path"]
    kmeans_filepath = parameters_json["Kmeans model path"]
    show_violin = bool(parameters_json["Show Violin"].lower() == "true")
    save_violin = bool(parameters_json["Save Violin"].lower() == "true")
    save_model = bool(parameters_json["Save Model"].lower() == "true")
    show_cluster_info = bool(parameters_json["Show cluster info"].lower() == "true")
    save_df = bool(parameters_json["Save dataframe"].lower() == "true")

    return df_filepath, cols, pop_comparison, pca_filepath, kmeans_filepath, show_violin, save_violin, \
           save_model, show_cluster_info, save_df


def apply_saved_model_and_analyse(config_file: str = "apply_model_parameters.json"):
    """
        Analysis of the dataset based on a config file. The previously trained Standard Scaler and PCA will be loaded
        and reused. A previously trained K-means will be loaded and reused.
        Parameters :
            config_file: path to a config file
    """
    df_filepath, cols, pop_comparison, pca_filepath, kmeans_filepath, show_violin, save_violin, \
    save_model, show_cluster_info, save_df = data_parser(config_file)
    if ".csv" in df_filepath:
        df = pd.read_csv(df_filepath)
    else:
        df = pd.read_excel(df_filepath)
    pca_pipeline = pickle.load(open(pca_filepath, "rb"))
    seed = pca_pipeline["pca"].random_state
    k_means = pickle.load(open(kmeans_filepath, "rb"))

    pca_cols = []
    pcs = pca_pipeline.transform(df[cols])
    for pc in range(1, pca_pipeline["pca"].n_components + 1):
        df[f"Principal component {pc}"] = pcs[:, pc - 1]
        pca_cols.append(f"Principal component {pc}")
    subpop_dict = {}
    k_means_dict = {}
    df["Clusters"] = k_means.predict(df[pca_cols])
    for subpop in np.unique(df[pop_comparison]):
        k_means_dict[subpop] = k_means
        extracted_pop_df = df[df[pop_comparison] == subpop]
        extracted_pop_df[f"{subpop} clusters"] = extracted_pop_df["Clusters"]
        subpop_dict[f"{subpop} clusters"] = extracted_pop_df
        if save_df:
            extracted_pop_df.to_csv(f"saved_analysis/{seed}/df_{pop_comparison}_{k_means_dict[subpop]}_2clusters.csv")
        if show_cluster_info:
            extracted_pop_df[cols] = extracted_pop_df[cols].astype(np.float64)
            print(subpop)
            print("Mean : \n", extracted_pop_df[cols + pca_cols].mean())
            print("Mean of each cluster : \n", extracted_pop_df.groupby([f"{subpop} clusters"]).mean()[cols + pca_cols])
            print(f"{subpop} {k_means_dict[subpop]} clusters", collections.Counter(list(extracted_pop_df["Clusters"])))

    if show_violin is True or save_violin is True:
        general_population_violin(subpop_dict, seed, cols, pca_cols, show_violin, save_violin)


if __name__ == '__main__':
    apply_saved_model_and_analyse()
