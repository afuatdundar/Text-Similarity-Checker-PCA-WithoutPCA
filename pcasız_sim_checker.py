import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import preprocess_texts, read_texts
from embedding import compute_embeddings

def print_similarity_scores(original_filename, original_embedding, altered_embeddings, altered_file_names):
    # Compute cosine similarities
    similarity_scores = cosine_similarity(original_embedding.reshape(1, -1), altered_embeddings).flatten()
    
    # Print similarity scores
    print(f"Similarity scores for {original_filename}:")
    for filename, score in zip(altered_file_names, similarity_scores):
        print(f"{filename}: {score:.4f}")

     # Print embeddings for each altered file
    print("\nEmbeddings for each altered file:")
    for filename, embedding in zip(altered_file_names, altered_embeddings):
        print(f"{filename}: {embedding[:10]}...")  # Print first 10 values for brevity

def main():
    # Script'in bulunduğu dizini alın
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Orijinal ve değiştirilmiş metinlerin bulunduğu dizinleri belirtin
    original_texts_directory = os.path.join(script_dir, "chosen_text")
    altered_texts_directory = os.path.join(script_dir, "samples")
    
    # Orijinal dizini kontrol edin
    if not os.path.exists(original_texts_directory):
        print(f"Error: Directory {original_texts_directory} does not exist.")
        return
    
    # Değiştirilmiş dizini kontrol edin
    if not os.path.exists(altered_texts_directory):
        print(f"Error: Directory {altered_texts_directory} does not exist.")
        return
    
    # Dosya adlarını al
    original_file_names = [f for f in os.listdir(original_texts_directory) if os.path.isfile(os.path.join(original_texts_directory, f))]
    altered_file_names = [f for f in os.listdir(altered_texts_directory) if os.path.isfile(os.path.join(altered_texts_directory, f))]
    
    # Orijinal metinleri okuyun ve temizleyin
    texts = read_texts(original_texts_directory)
    cleaned_texts = preprocess_texts(texts)
    
    # Compute embeddings for original texts
    embeddings = compute_embeddings(cleaned_texts)
    
    # Değiştirilmiş metinleri okuyun ve temizleyin
    altered_texts = read_texts(altered_texts_directory)
    cleaned_altered_texts = preprocess_texts(altered_texts)
    
    # Compute embeddings for altered texts
    altered_embeddings = compute_embeddings(cleaned_altered_texts)
    
    # Assume we compare the first original text with all altered texts
    original_index = 0  # Index of the original text to compare
    original_filename = original_file_names[original_index]
    original_embedding = embeddings[original_index]
    
    # Print similarity scores and embeddings
    print_similarity_scores(original_filename, original_embedding, altered_embeddings, altered_file_names)

if __name__ == "__main__":
    main()
