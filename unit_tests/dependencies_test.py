import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import os
import plotly.graph_objects as go


def test_pca():
    """
    Test if the PCA is giving the same result no matter the seed.
    """
    seed = 123456789
    df = pd.read_csv("general_population.csv")
    cols = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    x = df[cols]
    pca = PCA(n_components=3, random_state=seed, svd_solver="full")
    pca2 = PCA(n_components=3, random_state=0, svd_solver="full")
    pcs_1 = pca.fit_transform(x)
    pcs_2 = pca2.fit_transform(x)
    assert np.testing.assert_allclose(pcs_1, pcs_2) is None


def test_kmeans():
    """
    Test if kmeans is giving the same result
    """
    seed = 123456789
    df = pd.read_csv("general_population.csv")
    cols = ["feature1", "feature2", "feature3", "feature4", "feature5"]

    x = df[cols]
    pca = PCA(n_components=3, random_state=seed, svd_solver="full")
    pca.fit(x)
    principal_components = pca.transform(x)

    k_means = KMeans(n_clusters=4, random_state=seed).fit(principal_components)

    k_means2 = KMeans(n_clusters=4, random_state=seed).fit(principal_components)
    k_means2.predict(principal_components)
    k_means.predict(principal_components)
    assert np.testing.assert_array_equal(k_means.labels_, k_means2.labels_) is None


def test_pca_pickle():
    """
    Test if pickle has no influence on the PCA
    """
    seed = 123456789
    df = pd.read_csv("general_population.csv")
    cols = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    x = df[cols]
    pca = PCA(n_components=3, random_state=seed, svd_solver="full")
    pca.fit(x)
    pickle.dump(pca, open("pca.pkl", "wb"))
    pca2 = pickle.load(open("pca.pkl", "rb"))

    x_1 = pca.transform(x)
    x_2 = pca2.transform(x)

    assert np.testing.assert_allclose(x_1, x_2) is None


def test_kmeans_pickle():
    """
    Test if pickle has no influence on kmeans
    """
    seed = 123456789
    df = pd.read_csv("general_population.csv")
    cols = ["feature1", "feature2", "feature3", "feature4", "feature5"]

    x = df[cols]
    pca = PCA(n_components=3, random_state=seed, svd_solver="full")
    principal_components = pca.fit_transform(x)
    k_means = KMeans(n_clusters=4, random_state=seed).fit(principal_components)
    pickle.dump(k_means, open("kmeans.pkl", "wb"))

    k_means2 = pickle.load(open("kmeans.pkl", "rb"))
    os.remove("kmeans.pkl")
    k_means2.predict(principal_components)
    assert np.testing.assert_allclose(k_means.labels_, k_means2.labels_) is None


def test_pipeline():
    """
    Test that pipeline is giving the same results has not using pipeline
    """
    seed = 123456789
    model = make_pipeline(StandardScaler(),
                          PCA(n_components=3, random_state=seed, svd_solver="full"),
                          )
    df = pd.read_csv("general_population.csv")
    
    cols = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    x = df[cols]
    model.fit(x)
    principalcomponents = model.transform(x)

    scaled_x = StandardScaler().fit_transform(x)
    principalcomponents2 = PCA(n_components=3, random_state=seed, svd_solver="full").fit_transform(scaled_x)
    assert np.testing.assert_allclose(principalcomponents, principalcomponents2) is None


def test_pipeline_pickle():
    """
        Test if pickle has no influence on pipeline
    """
    seed = 123456789
    model = make_pipeline(StandardScaler(),
                          PCA(n_components=3, random_state=seed, svd_solver="full"),
                          )
    df = pd.read_csv("general_population.csv")

    cols = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    x = df[cols]
    model.fit(x)
    pickle.dump(model, open("model.pkl", "wb"))
    principalcomponents = model.transform(x)

    model2 = pickle.load(open("model.pkl", "rb"))
    os.remove("model.pkl")

    principalcomponents2 = model2.transform(x)
    assert np.testing.assert_allclose(principalcomponents, principalcomponents2) is None
