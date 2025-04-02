import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import sklearn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from opensoundscape.ml import bioacoustics_model_zoo as bmz
from opensoundscape.ml.shallow_classifier import quick_fit

from scipy.special import softmax
from matplotlib import pyplot as plt
from collections import defaultdict

# from matplotlib import pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm


class DataProcessor:
    def __init__(self, datapath, species_list, datatype="data"):
        self.datapath = datapath
        self.species_list = species_list
        self.datatype = "data"
        self.all_species = pd.DataFrame()
        self.df_each_species = defaultdict(list)

    def load_data(self):
        """Load file paths and labels into a structured DataFrame."""
        print("Loading dataset...") 
        for species in self.species_list:
            pos_files = glob.glob(
                os.path.join(self.datapath, species, self.datatype, "pos", "*.wav")
            )
            neg_files = glob.glob(
                os.path.join(self.datapath, species, self.datatype, "neg", "*.wav")
            )
            all_files = pos_files + neg_files

            encoding_pos_files = [1] * len(pos_files) + [0] * len(neg_files)
            encoding_neg_files = [0] * len(pos_files) + [1] * len(neg_files)

            df_species = pd.DataFrame(
                {
                    "files": all_files,
                    species: encoding_pos_files,
                    "noise": encoding_neg_files,
                }
            )
            self.df_each_species[species] = df_species

        self.all_species = pd.concat(self.df_each_species.values(), axis=0)
        self.all_species.fillna(0, inplace=True)
        self.all_species.set_index("files", inplace=True)
        self.all_species = self.all_species.astype(int)
        print("Dataset loaded:")
        print(self.all_species.sum('index'))  # ADDED
        print(self.all_species.head())
        return self.all_species


class Model:
    def __init__(self, model_name="BirdNET"):
        print("Loading model...")
        self.model = torch.hub.load("kitzeslab/bioacoustics-model-zoo", "BirdNET", trust_repo=True)
        self.num_workers = os.cpu_count() * 3 // 4  # Use 75% of CPU cores
        print(f"CPU CORES: {self.num_workers}")
        print(f"Model loaded: {model_name}")

    def train_and_evaluate(self, df, species_list, folds=5):
        """Perform stratified K-fold training and evaluation."""
        self.species_list = species_list
        self.all_scores = defaultdict(list)
        
        for species in self.species_list:
            print(f"Processing species: {species}")

            file_paths = df.index
            labels = df[species]
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=8)
            roc_auc_scores = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(file_paths, labels)):
                
                train_files, test_files = (file_paths[train_idx].tolist(), file_paths[test_idx].tolist())
                labels_train, labels_val = labels.iloc[train_idx], labels.iloc[test_idx]

                labels_train = labels_train.to_numpy().reshape(-1, 1)
                labels_val = labels_val.to_numpy().reshape(-1, 1)

                emb_train = self.model.embed(train_files, return_dfs=False, batch_size=4, num_workers=self.num_workers)
                emb_val = self.model.embed(test_files, return_dfs=False, batch_size=4, num_workers=self.num_workers)

                self.model.change_classes([species])
                self.model.network.fit(emb_train, labels_train, emb_val, labels_val)

                preds = self.model.network(torch.tensor(emb_val)).detach().numpy()
                curr_score = roc_auc_score(labels_val, preds, average=None)
                roc_auc_scores.append(curr_score)

                print(f"Fold {fold_idx + 1}: ROC AUC Score = {curr_score}")

            avg_roc_auc = np.mean(roc_auc_scores)
            print(f"Average Across All Folds: {avg_roc_auc}")
            all_scores[species] = avg_roc_auc

        print(f"All scores: {all_scores}")
        return all_scores