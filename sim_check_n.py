import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from clustering import reduce_dimensions
from data_processing import preprocess_texts, read_texts
from embedding import compute_embeddings
from sklearn.decomposition import PCA

def plot_explained_variance(embeddings):
    pca = PCA().fit(embeddings)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.axhline(y=0.90, color='r', linestyle='-')
    plt.axhline(y=0.95, color='g', linestyle='-')
    plt.grid(True)
    plt.show()

def get_file_names(directory):
    """ Read text file names from a directory """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def plot_similarity_matrix(matrix, labels, title):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def print_file_names(labels):
    """ Print file names for copying """
    print("\nFile names:")
    for label in labels:
        print(label)

def compute_and_plot_similarity_matrices(embeddings, pca_result, original_file_names, altered_embeddings, altered_pca_result, altered_file_names):
    # Compute cosine similarities for embeddings
    cos_sim_matrix_embeddings = cosine_similarity(embeddings)
    # Compute cosine similarities for PCA results
    cos_sim_matrix_pca = cosine_similarity(pca_result)
    
    # Compute cosine similarities for embeddings and PCA results with altered texts
    cos_sim_matrix_embeddings_altered = cosine_similarity(embeddings, altered_embeddings)
    cos_sim_matrix_pca_altered = cosine_similarity(pca_result, altered_pca_result)
    
    # Plot similarity matrices for original texts
    plot_similarity_matrix(cos_sim_matrix_embeddings, original_file_names, 'Similarity Matrix (Embeddings)')
    plot_similarity_matrix(cos_sim_matrix_pca, original_file_names, 'Similarity Matrix (PCA)')
    
    # Create and plot similarity matrices for original vs altered texts
    num_orig = len(original_file_names)
    num_alt = len(altered_file_names)
    
    # Create empty matrices for visualization
    cos_sim_matrix_embeddings_altered_full = np.zeros((num_orig, num_alt))
    cos_sim_matrix_pca_altered_full = np.zeros((num_orig, num_alt))
    
    # Fill in the matrices with the corresponding similarity values
    cos_sim_matrix_embeddings_altered_full[:num_orig, :num_alt] = cos_sim_matrix_embeddings_altered
    cos_sim_matrix_pca_altered_full[:num_orig, :num_alt] = cos_sim_matrix_pca_altered
    
    # Create labels for the combined matrices
    orig_labels = original_file_names
    alt_labels = altered_file_names
    combined_labels = orig_labels + alt_labels
    
    # Plot the combined similarity matrices
    plot_similarity_matrix(cos_sim_matrix_embeddings_altered_full, alt_labels, 'Similarity Matrix (Embeddings vs Altered)')
    plot_similarity_matrix(cos_sim_matrix_pca_altered_full, alt_labels, 'Similarity Matrix (PCA vs Altered)')
    
    # Print file names for copying
    print_file_names(orig_labels + alt_labels)

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
    
    # Plot explained variance to determine optimal number of components
    plot_explained_variance(embeddings)
    
    # Compute PCA
    pca_n_components = int(input("Enter the number of PCA components: "))
    pca_result = reduce_dimensions(embeddings, n_components=pca_n_components)
    
    # Plot explained variance to determine optimal number of components
    plot_explained_variance(altered_embeddings)

    pca_n_components = int(input("Enter the number of PCA components: "))
    altered_pca_result = reduce_dimensions(altered_embeddings, n_components=pca_n_components)
    


    # Compute and plot similarity matrices
    compute_and_plot_similarity_matrices(embeddings, pca_result, original_file_names, altered_embeddings, altered_pca_result, altered_file_names)

if __name__ == "__main__":
    main()
