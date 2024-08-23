import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import preprocess_texts, read_texts
from embedding import compute_embeddings

def plot_explained_variance(embeddings):
    pca = PCA().fit(embeddings)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Explained Variance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.show()

def apply_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

def compute_similarity_scores(original_embedding, altered_embeddings, pca):
    # Apply PCA to original embedding and altered embeddings
    pca_altered_embeddings = pca.transform(altered_embeddings)
    
    # Compute cosine similarities
    similarity_scores = cosine_similarity(original_embedding.reshape(1, -1), pca_altered_embeddings).flatten()
    return similarity_scores

def print_similarity_scores(original_filename, original_embedding, altered_embeddings, altered_file_names, pca):
    # Apply PCA to original embedding
    pca_original_embedding = pca.transform(original_embedding.reshape(1, -1)).flatten()
    
    # Compute similarity scores
    similarity_scores = compute_similarity_scores(pca_original_embedding, altered_embeddings, pca)
    
    # Print similarity scores
    print(f"Similarity scores for {original_filename}:")
    for filename, score in zip(altered_file_names, similarity_scores):
        print(f"{filename}: {score:.4f}")

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
    print(f"altered_texts Count: {len(altered_texts)}")
    cleaned_altered_texts = preprocess_texts(altered_texts)
    print(f"cleaned_altered_texts Count: {len(cleaned_altered_texts)}")
    # Compute embeddings for altered texts
    altered_embeddings = compute_embeddings(cleaned_altered_texts)
    
    # Plot explained variance to determine optimal number of PCA components for altered texts
    plot_explained_variance(altered_embeddings)
    
    # Enter PCA component number based on the plot
    pca_n_components = int(input("Enter the number of PCA components for altered texts: "))
    
    # Apply PCA to altered embeddings
    pca = PCA(n_components=pca_n_components)
    pca.fit(altered_embeddings)
    
    # Assume we compare the first original text with all altered texts
    original_index = 0  # Index of the original text to compare
    original_filename = original_file_names[original_index]
    original_embedding = embeddings[original_index]
    
    # Print similarity scores using PCA on altered texts
    print_similarity_scores(original_filename, original_embedding, altered_embeddings, altered_file_names, pca)

    print(f"Original File Count: {len(original_file_names)}")
    print(f"Original Embeddings Count: {len(embeddings)}")
    print(f"Altered File Count: {len(altered_file_names)}")
    print(f"Altered Embeddings Count: {len(altered_embeddings)}")

    print(f"Original File Names: {original_file_names}")
    print(f"Altered File Names: {altered_file_names}")


if __name__ == "__main__":
    main()
