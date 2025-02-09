import gensim
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Word2Vec vectors location
#TODO change it so it downloads automatically
location = r'C:\Users\Doge\Downloads\GoogleNews-vectors-negative300.bin.gz'


#wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000)
#wv.save_word2vec_format('vectors.csv')

# Check the csv
with open('vectors.csv', 'r', encoding='utf-8') as x:
    for i in range(5):
        print(x.readline())

# CSV is delimited by spaces, have to skip first row
vectors = pd.read_csv('vectors.csv', delimiter=' ', header=None, skiprows=1, encoding='ISO-8859-1')
print("vectors")
print(vectors.head())

# Convert vectors into dictionary
word_keys = vectors.iloc[:, 0].values
word_vectors = vectors.iloc[:, 1:].values.astype(np.float32)
v_dict = dict(zip(word_keys, word_vectors))
v_dict = {str(key).lower(): value for key, value in v_dict.items() if isinstance(key, str)}

# UTF-8 fails, have to use ISO-8859-1
phrases_df = pd.read_csv('phrases.csv', header = None, encoding="ISO-8859-1")
phrases = phrases_df.iloc[1:, 0].astype(str).tolist()
print("phrases")
print(phrases_df.head())

# a) Assign each word in each phrase a Word2Vec embedding.
def assign_embeddings(phrases, v_dict):
    phrase_embeddings = []
    
    for phrase in phrases:
        words = phrase.lower().split()
        embeddings = [v_dict[word] for word in words if word in v_dict]
        
        if embeddings:
            phrase_embeddings.append((phrase, np.array(embeddings)))
        else:
            phrase_embeddings.append((phrase, None))

    return phrase_embeddings

# Assign embeddings
phrase_embeddings = assign_embeddings(phrases, v_dict)

# Convert to DataFrame for better readability
phrase_embeddings_df = pd.DataFrame([
    (phrase, emb.shape if emb is not None else "No embeddings found")
    for phrase, emb in phrase_embeddings
], columns=["Phrase", "Embedding Shape"])

# Display first 5 results
print("phrase_embeddings_df")
print("-------------------")
print(phrase_embeddings_df.head())
print("-------------------")


# Check the result
for phrase, embeddings in phrase_embeddings[:5]:
    print(f"Phrase: {phrase}")
    if isinstance(embeddings, np.ndarray):
        print(f"Embedding Shape: {embeddings.shape}")
    else:
        print("No embeddings found")
    print("---------------------------------------------------------")

# b) Batch execution: Calculate L2 distance (Euclidean distance) or Cosine distance of each phrase to all other phrases and store results.

def compute_phrase_vectors(phrase_embeddings):
    phrase_vectors = []
    for phrase, embeddings in phrase_embeddings:
        if isinstance(embeddings, np.ndarray):
            phrase_vector = np.mean(embeddings, axis=0)
        else:
            phrase_vector = np.zeros(300, dtype=np.float32)
        
        phrase_vectors.append(phrase_vector)
    
    return np.vstack(phrase_vectors)

# Phrase vectors
phrase_vectors = compute_phrase_vectors(phrase_embeddings)

def batch_distance_calculation(phrase_vectors, metric="euclidean", batch_size=5):
    num_phrases = phrase_vectors.shape[0]
    distance_matrix = np.zeros((num_phrases, num_phrases), dtype=np.float32)

    for start in range(0, num_phrases, batch_size):
        end = min(start + batch_size, num_phrases)
        # Distance between current batch and other phrases
        distances = cdist(phrase_vectors[start:end], phrase_vectors, metric=metric)
        distance_matrix[start:end, :] = distances
    return distance_matrix

# Compute distances (L2 or Cosine)
l2_distances = batch_distance_calculation(phrase_vectors, metric="euclidean")
cosine_distances = batch_distance_calculation(phrase_vectors, metric="cosine")

# Convert to DataFrame, check results
phrases_list = [phrase for phrase, _ in phrase_embeddings]
l2_df = pd.DataFrame(l2_distances, index=phrases_list, columns=phrases_list)
print("----------------------------------------")
print("L2 distance")
print(l2_df.head())
print("----------------------------------------")
cosine_df = pd.DataFrame(cosine_distances, index=phrases_list, columns=phrases_list)
print("----------------------------------------")
print("Cosine distance")
print(cosine_df.head())
print("----------------------------------------")

# c) On the fly execution: Create a function that takes any string, e.g. user-input phrase, and finds and return the closest match from phrases in phrases.csv and the distance

def find_closest_match(input_phrase, phrases, phrase_vectors, v_dict, metric="euclidean"):
    words = input_phrase.lower().split()
    embeddings = [v_dict[word] for word in words if word in v_dict]
    if embeddings:
        input_vector = np.mean(embeddings, axis=0).reshape(1, -1)
    else:
        return None, None