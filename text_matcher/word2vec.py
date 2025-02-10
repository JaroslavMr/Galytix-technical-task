import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

class Word2vec:
    def __init__(self, input_file="GoogleNews-vectors-negative300.bin.gz", output_file="vectors.csv"):
        self.model_path = input_file
        self.vector_file = output_file
        self.v_dict = {}

    def load_word2vec(self):
        try:
            if not os.path.exists(self.vector_file):
                wv = KeyedVectors.load_word2vec_format(self.model_path, binary=True, limit=1000000)
                wv.save_word2vec_format(self.vector_file)

            # Read vectors from CSV
            vectors = pd.read_csv(self.vector_file, delimiter=' ', header=None, skiprows=1, encoding='ISO-8859-1')
            word_keys = vectors.iloc[:, 0].values
            word_vectors = vectors.iloc[:, 1:].values.astype(np.float32)
            self.v_dict = {str(key).lower(): value for key, value in zip(word_keys, word_vectors)}

        except Exception as e:
            raise

    def get_vector_dict(self):
        return self.v_dict