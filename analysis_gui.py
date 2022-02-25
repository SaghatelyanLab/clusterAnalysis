from tkinter import filedialog
import pandas as pd
from tkinter import *

import json


class MyWindow:
    dataset = None

    def __init__(self, win):
        # self.win = win
        self.save_violin = BooleanVar()
        self.show_violin = None
        self.save_violin_cb = None
        self.generate_jsonButton = None
        self.dataset_path = None
        self.widget_label_list = []
        self.widget_entry_list = []
        self.mainGrid = Grid()
        self.b1 = Button(win, text='Load dataset', command=self.load_dataset)
        self.b1.grid(column=1, row=0, pady=10, padx=10, sticky="ew")


    def load_dataset(self):
        self.dataset, self.dataset_path = load_dataset()
        self.instructions = Label(window, text="Select feature columns\n and column for comparison\n then click on K-means.")
        self.instructions.grid(column=1, row=1, pady=10, padx=10)

        self.frame = Frame(window)
        self.frame.grid(column=0, row=2, pady=10, padx=10)

        self.featureColumnsLabel = Label(self.frame, text="Feature columns")
        self.featureColumns = Listbox(self.frame, width=20, height=6, selectmode="multiple",
                                      exportselection=False)
        self.featureColumnsLabel.pack(side="top")
        self.featureColumns.pack(side='left', fill='y')

        scrollbar = Scrollbar(self.frame, orient="vertical", command=self.featureColumns.yview)
        scrollbar.pack(side="right", fill="y")

        self.featureColumns.config(yscrollcommand=scrollbar.set)

        self.frame2 = Frame(window)
        self.frame2.grid(column=1, row=2, pady=10, padx=10)

        self.comparisonColumnLabel = Label(self.frame2, text="Comparison column")
        self.comparisonColumn = Listbox(self.frame2, width=20, height=6, selectmode="BROWSE",
                                        exportselection=False)
        self.comparisonColumnLabel.pack(side="top", fill="y")
        self.comparisonColumn.pack(side='left', fill='y')

        scrollbar2 = Scrollbar(self.frame2, orient="vertical", command=self.comparisonColumn.yview)
        scrollbar2.pack(side="right", fill="y")

        self.comparisonColumn.config(yscrollcommand=scrollbar2.set)

        for column in self.dataset.columns:
            self.featureColumns.insert(END, column)
            self.comparisonColumn.insert(END, column)

        # self.dataset, self.dataset_path = load_dataset()
        # yscrollbar = Scrollbar(window)
        # yscrollbar.grid(column=2, row=2, pady=10, padx=10)
        # yscrollbar2 = Scrollbar(window)
        # yscrollbar2.grid(column=4, row=2, pady=10, padx=10)
        # self.featureColumns = Listbox(window, selectmode="multiple",
        #                               yscrollcommand=yscrollbar.set)
        # self.featureColumns.grid(column=1, row=2, pady=10, padx=10)
        # self.comparisonColumn = Listbox(window, selectmode="BROWSE",
        #                                 yscrollcommand=yscrollbar2.set)
        # self.comparisonColumn.grid(column=3, row=2, pady=10, padx=10)
        #
        # for column in self.dataset.columns:
        #     self.featureColumns.insert(END, column)
        #     self.comparisonColumn.insert(END, column)
        self.KmeansButton = Button(window, text="K-means", command=self.kmeans)
        self.KmeansButton.grid(column=1, row=3, pady=10, padx=10)

    def kmeans(self):
        if len(self.widget_label_list) >= 1:
            for widget in self.widget_label_list:
                widget.destroy()
            for widget in self.widget_entry_list:
                widget.destroy()
        for i, pop in enumerate(self.dataset[self.comparisonColumn.get(ANCHOR)].unique()):
            popLabel = Label(window, text=pop)
            popEntry = Entry(window)
            popEntry.insert(END, "0")
            self.widget_label_list.append(popLabel)
            self.widget_entry_list.append(popEntry)
            popLabel.grid(row=4, column=i, pady=10, padx=10)
            popEntry.grid(row=5, column=i, pady=10, padx=10)

        self.KmeansLabel = Label(window, text="Enter number of clusters\n for each population.")
        self.KmeansLabel.grid(column=0, row=3, pady=10, padx=10)

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
        self.seedEntry.insert(END, 'RANDOM')

        self.pcLabel = Label(window, text="Number of principal components")
        self.pcEntry = Entry(window)

        self.seedLabel.grid(row=7, column=0, pady=10, padx=10)
        self.seedEntry.grid(row=8, column=0, pady=10, padx=10)
        self.pcLabel.grid(row=7, column=2, pady=10, padx=10)
        self.pcEntry.grid(row=8, column=2, pady=10, padx=10)

        self.generate_jsonButton = Button(window, text="Generate json parameters", command=self.generate_json)
        self.analyseButton = Button(window, text="Start analysis", command=self.start_analysis)
        self.generate_jsonButton.grid(row=9, column=1, pady=10, padx=10)
        self.analyseButton.grid(row=9, column=2, pady=10, padx=10)

    def generate_data_dict(self):
        KmeansDict = {}
        for label, entry in zip(self.widget_label_list, self.widget_entry_list):
            KmeansDict[label.cget("text")] = entry.get()
        feature_values = [self.featureColumns.get(idx) for idx in self.featureColumns.curselection()]

        self.data = {"DataSetPath": self.dataset_path,
                     "FeaturesColumns": feature_values,
                     "ComparisonColumn": self.comparisonColumn.get(self.comparisonColumn.curselection()),
                     "Seed": self.seedEntry.get(),
                     "Number of PCs": self.pcEntry.get(),
                     "Kmeans": KmeansDict,
                     "Show Violin": self.show_violin.get(),
                     "Save Violin": self.save_violin.get(),
                     "Save Model": self.save_model.get(),
                     "Show cluster info": self.show_cluster_info.get(),
                     "Save dataframe": self.save_df.get()}

    def start_analysis(self):
        self.generate_data_dict()
        json.dump(self.data, open("general_analysis_parameters.json", "w"), indent=4)
        window.destroy()
        from analysis import general_analysis
        general_analysis("general_analysis_parameters.json")

    def generate_json(self):
        self.generate_data_dict()
        json.dump(self.data, open("general_analysis_parameters.json", "w"), indent=4)
        from tkinter import messagebox
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


def convert_rgb(rgb):
    return "#%02x%02x%02x" % rgb


if __name__ == '__main__':
    window = Tk()
    mywin = MyWindow(window)
    window.title('Cluster Analysis')
    window.geometry("800x600+10+10")
    # window.grid_columnconfigure((0, 1, 2), weight=1)
    # window.config(bg=convert_rgb("white smoke"))
    # window.config(bg="white smoke")
    window.mainloop()
