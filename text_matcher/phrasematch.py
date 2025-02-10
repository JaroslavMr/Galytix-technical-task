import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class PhraseMatch:
    def __init__(self, v_dict, phrases_file="phrases.csv"):
        self.v_dict = v_dict
        self.phrases = []
        self.phrase_vectors = None
        self.load_phrases(phrases_file)

    def load_phrases(self, phrases_file):
        phrases_df = pd.read_csv(phrases_file, header=None, encoding="ISO-8859-1")
        self.phrases = phrases_df.iloc[1:, 0].astype(str).tolist()
        self.assign_embeddings()
    """
    def assign_word_embeddings(self, phrases):
        phrase_embeddings = []
        
        for phrase in phrases:
            words = phrase.lower().split()
            embeddings = [self.v_dict[word] for word in words if word in self.v_dict]
            
            if embeddings:
                phrase_embeddings.append((phrase, np.array(embeddings)))
            else:
                phrase_embeddings.append((phrase, None))

        return phrase_embeddings
    """
    
    def assign_embeddings(self):
        self.phrase_embeddings = [
            (phrase, np.array([self.v_dict[word] for word in phrase.lower().split() if word in self.v_dict]))
            for phrase in self.phrases
        ]
        self.compute_phrase_vectors()


    def compute_phrase_vectors(self):
        self.phrase_vectors = np.vstack([
            np.mean(embeddings, axis=0) if embeddings.size else np.zeros(300)
            for _, embeddings in self.phrase_embeddings
        ])

    def find_closest_match(self, input_phrase, metric_input):
        words = input_phrase.lower().split()
        metric =  metric_input
        embeddings = np.array([self.v_dict[word] for word in words if word in self.v_dict])

        if embeddings.size == 0:

            return None, None

        input_vector = np.mean(embeddings, axis=0).reshape(1, -1)
        distances = cdist(input_vector, self.phrase_vectors, metric=metric)
        min_index = np.argmin(distances)

        return self.phrases[min_index], distances[0, min_index]

