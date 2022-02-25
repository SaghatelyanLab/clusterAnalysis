import pickle
from tkinter import filedialog
import pandas as pd
from tkinter import *
from tkinter import messagebox

import json


class MyWindow:
    dataset = None

    def __init__(self, win):
        self.save_violin = BooleanVar()
        self.show_violin = None
        self.save_violin_cb = None
        self.generate_jsonButton = None
        self.dataset_path = None
        self.widget_label_list = []
        self.widget_entry_list = []
        self.mainGrid = Grid()
        self.dataset_filename = None
        self.kmeans_filename = None
        self.pca_filename = None
        self.dataset_exists = False
        self.kmeans_exists = False
        self.pca_exists = False
        self.data = None
        self.b1 = Button(win, text='Load new dataset', command=self.load_dataset)
        self.b1.grid(column=1, row=1, pady=10, padx=10)
        self.b2 = Button(win, text='Load trained PCA', command=self.load_PCA)
        self.b2.grid(column=1, row=4, pady=10, padx=10)
        self.b3 = Button(win, text='Load trained Kmeans', command=self.load_kmeans)
        self.b3.grid(column=2, row=4, pady=10, padx=10)

    def load_dataset(self):
        self.dataset, self.dataset_path = load_dataset()
        self.frame = Frame(window)
        self.frame.grid(column=1, row=2, pady=10, padx=10)

        self.featureColumns = Listbox(self.frame, width=20, height=6, selectmode="multiple",
                                      exportselection=False)
        self.featureColumns.pack(side='left', fill='y')

        scrollbar = Scrollbar(self.frame, orient="vertical", command=self.featureColumns.yview)
        scrollbar.pack(side="right", fill="y")

        self.featureColumns.config(yscrollcommand=scrollbar.set)

        self.frame2 = Frame(window)
        self.frame2.grid(column=2, row=2, pady=10, padx=10)

        self.comparisonColumn = Listbox(self.frame2, width=20, height=6, selectmode="BROWSE",
                                        exportselection=False)
        self.comparisonColumn.pack(side='left', fill='y')

        scrollbar2 = Scrollbar(self.frame2, orient="vertical", command=self.comparisonColumn.yview)
        scrollbar2.pack(side="right", fill="y")

        self.comparisonColumn.config(yscrollcommand=scrollbar2.set)

        for column in self.dataset.columns:
            self.featureColumns.insert(END, column)
            self.comparisonColumn.insert(END, column)

        self.dataset_exists = True
        self.dataset_filename = self.dataset_path
        self.check_for_all_data()

    def load_kmeans(self):
        kmeansObject, kmeans_filename = load_pickle("Kmeans")
        self.KmeansLabel = Label(window, text=f"Number of target clusters : {verify_kmeans(kmeansObject)}")
        self.KmeansLabel.grid(column=2, row=5)
        self.kmeans_filename = kmeans_filename
        self.kmeans_exists = True
        self.check_for_all_data()

    def load_PCA(self):
        pcaObject, pca_filename = load_pickle("PCA")
        self.PCALabel = Label(window, text=f"Number of Principal components : {verify_PCA(pcaObject)}")
        self.PCALabel.grid(column=1, row=5)
        self.pca_filename = pca_filename
        self.pca_exists = True
        self.check_for_all_data()

    def check_for_all_data(self):
        if self.dataset_filename is not None and self.pca_filename is not None and self.kmeans_filename is not None:
            self.show_violin = BooleanVar(value=True)
            self.show_violin_cb = Checkbutton(window, text='Show violin', variable=self.show_violin)
            self.show_violin_cb.grid(row=6, column=0, pady=10, padx=10)
            self.save_violin = BooleanVar(value=True)
            self.save_violin_cb = Checkbutton(window, text='Save Violin', variable=self.save_violin)
            self.save_violin_cb.grid(row=6, column=1, pady=10, padx=10)
            self.save_model = BooleanVar(value=True)
            self.save_model_cb = Checkbutton(window, text='Save Model', variable=self.save_model)
            self.save_model_cb.grid(row=6, column=2, pady=10, padx=10)
            self.show_cluster_info = BooleanVar(value=True)
            self.show_cluster_info_cb = Checkbutton(window, text='Show cluster info', variable=self.show_cluster_info)
            self.show_cluster_info_cb.grid(row=6, column=3, pady=10, padx=10)
            self.save_df = BooleanVar(value=True)
            self.save_df_cb = Checkbutton(window, text='Save dataframe', variable=self.save_df)
            self.save_df_cb.grid(row=6, column=4, pady=10, padx=10)

            self.seedLabel = Label(window, text="Seed")
            self.seedEntry = Entry(window)

            self.generate_jsonButton = Button(window, text="Generate json parameters", command=self.generate_json)
            self.analyseButton = Button(window, text="Start analysis", command=self.start_analysis)
            self.generate_jsonButton.grid(row=7, column=1, pady=10, padx=10)
            self.analyseButton.grid(row=7, column=2, pady=10, padx=10)

    def generate_data_dict(self):
        feature_values = [self.featureColumns.get(idx) for idx in self.featureColumns.curselection()]

        self.data = {"DataSetPath": self.dataset_path,
                     "FeaturesColumns": feature_values,
                     "ComparisonColumn": self.comparisonColumn.get(self.comparisonColumn.curselection()),
                     "PCA model path": self.pca_filename,
                     "Kmeans model path": self.kmeans_filename,
                     "Show Violin": self.show_violin.get(),
                     "Save Violin": self.save_violin.get(),
                     "Save Model": self.save_model.get(),
                     "Show cluster info": self.show_cluster_info.get(),
                     "Save dataframe": self.save_df.get()}

    def start_analysis(self):
        self.generate_data_dict()
        json.dump(self.data, open("apply_model_parameters.json", "w"), indent=4)
        from apply_saved_model import apply_saved_model_and_analyse
        apply_saved_model_and_analyse("apply_model_parameters.json")
        messagebox.showinfo("Cluster Analysis GUI", "Json filed had been created")

    def generate_json(self):
        self.generate_data_dict()
        json.dump(self.data, open("apply_model_parameters.json", "w"), indent=4)
        messagebox.showinfo("Cluster Analysis GUI", "Json filed had been created.")


def load_dataset():
    df_filename = filedialog.askopenfilename(title="Choose Dataset File",
                                             filetypes=[("All dataset files", "*.xls *.xlsx *.csv"),
                                                        ("Comma Separated Values", "*.csv"),
                                                        ("Excel files", "*.xls *.xlsx"),
                                                        ("XLS files", "*.xls"),
                                                        ("XLSX files", "*.xlsx"),
                                                        ])

    if ".xls" in df_filename or ".xlsx" in df_filename:
        df = pd.read_excel(df_filename)
    else:
        df = pd.read_csv(df_filename)
    return df, df_filename


def load_pickle(name=""):
    pickle_filename = filedialog.askopenfilename(title=f"Choose {name} File",
                                                 filetypes=[("Pickle file", "*.pkl")])

    return pickle.load(open(pickle_filename, "rb")), pickle_filename


def verify_kmeans(Kmeans):
    try:
        clusterNumber = getattr(Kmeans, "n_clusters")
        return clusterNumber
    except AttributeError:
        from tkinter import messagebox
        messagebox.showerror("Error when loading Kmeans object",
                             "This file seems to not be a valid Kmeans. Please load "
                             "a valid Kmeans pickle.")


def verify_PCA(pipeline):
    try:
        pcaNumber = getattr(pipeline["pca"], "n_components")
        return pcaNumber
    except AttributeError:
        from tkinter import messagebox
        messagebox.showerror("Error when loading PCA object", "This file seems to not be a valid PCA. Please load "
                                                              "a valid PCA pickle.")
    except TypeError:
        from tkinter import messagebox
        messagebox.showerror("Error when loading PCA object", "This file seems to not be a valid PCA but seems to be a "
                                                              "Kmeans pickle, please load a valid PCA pickle.")


if __name__ == '__main__':
    window = Tk()
    mywin = MyWindow(window)
    window.title('Cluster Analysis')
    window.geometry("800x600+10+10")
    window.mainloop()
