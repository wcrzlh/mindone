import os

import numpy as np
import pandas as pd
import random


class Text2ImageCacheDataset:
    def __init__(self, data_path, cache_path, multi_aspect=None):
        super().__init__()
        self.dataset_column_names = ["latent", "vector", "crossattn"]
        self.dataset_output_column_names = ["latent", "vector", "crossattn"]

        all_files = self.list_cache_files_recursively(data_path)
        self.all_files = all_files
        self.cache_path = cache_path
        self.multi_aspect = multi_aspect

    def __getitem__(self, idx):
        # images preprocess
        file_name = self.all_files[idx]

        if self.multi_aspect:
            file_pick = random.randint(0,self.multi_aspect-1)
            latent_path = os.path.join(self.cache_path, "latent_cache", str(file_pick)+".npy")
        else:
            latent_path = os.path.join(self.cache_path, "latent_cache", file_name)
        vector_path = os.path.join(self.cache_path, "vector_cache", file_name)
        crossattn_path = os.path.join(self.cache_path, "crossattn_cache", file_name)

        latent = np.load(latent_path).astype(np.float32)
        vector = np.load(vector_path).astype(np.float32)
        crossattn = np.load(crossattn_path).astype(np.float32)

        return latent, vector, crossattn

    def collate_fn(self, latents, vectors, crossattns, batch_info):
        if self.multi_aspect and len(latents)>1:
            latent_rp = latents[0]
            latents = [latent_rp] * len(latents)
        batch_latent = np.concatenate(latents, 0)
        batch_vector = np.concatenate(vectors, 0)
        batch_crossattn = np.concatenate(crossattns, 0)

        return batch_latent, batch_vector, batch_crossattn

    def __len__(self):
        return len(self.all_files)

    @staticmethod
    def list_cache_files_recursively(data_path):
        anno_dir = data_path
        anno_list = sorted(
            [os.path.join(anno_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(anno_dir)))]
        )
        db_list = [pd.read_csv(f) for f in anno_list]
        all_files = []
        for db in db_list:
            all_files.extend(list(db["dir"]))
        all_files = [f.split(".")[0] + ".npy" for f in all_files]

        return all_files
