import pandas as pd
import numpy as np
from text_matcher.word2vec import Word2vec
from text_matcher.phrasematch import PhraseMatch
from scipy.spatial.distance import cdist

def main():
    try:
        word2vec_loader = Word2vec()
        word2vec_loader.load_word2vec()
        v_dict = word2vec_loader.get_vector_dict()
        phrasematch = PhraseMatch(v_dict)
        phrases_df = pd.read_csv("phrases.csv", header=None, encoding="ISO-8859-1")
        phrases = phrases_df.iloc[1:, 0].astype(str).tolist()

# results of a) and b) can be checked in interactive window       
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
        phrasematch = PhraseMatch(v_dict)
        if phrasematch.phrase_vectors is None or len(phrasematch.phrase_vectors) == 0:
            return
        phrase_input = input("Enter a phrase: ").strip()
        metric_input = input("Euclidean or Cosine distance?: ").strip()
        # Check validity
        if metric_input not in ["euclidean", "cosine"]:
            print("Invalid metric! Defaulting to 'Euclidean'.")
            metric_input = "euclidean"

        best_match, distance = phrasematch.find_closest_match(phrase_input, metric_input)
        if best_match:
            print(f"\nClosest phrase: \"{best_match}\"")
            print(f"Distance ({metric_input}): {distance:.5f}")
        else:
            print("No match found.")

    except Exception as e:
        print(f"There was an error: {e}")

if __name__ == "__main__":
    main()
