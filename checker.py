import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from clustering import reduce_dimensions, cluster_texts, compute_cosine_distances, most_dissimilar_subset
from data_processing import read_texts, preprocess_texts
from embedding import compute_embeddings
from plotting import reduce_and_plot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

nltk.download('punkt')

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

def compute_and_print_variance(embeddings, pca_result):
    # Compute the variance of the original embeddings
    original_variance = np.var(embeddings, axis=0)
    total_original_variance = np.sum(original_variance)
    print(f"Total Original Variance: {total_original_variance}")

    # Compute the variance of the PCA-reduced embeddings
    pca_variance = np.var(pca_result, axis=0)
    total_pca_variance = np.sum(pca_variance)
    print(f"Total PCA Variance: {total_pca_variance}")

def main():
    output_dir = r"C:\Users\Asus\Desktop\project2\results"

    print("Starting data processing...")
    # Metin dosyalarını okuma ve ön işleme
    directory = r"C:\Users\Asus\Desktop\samples"
    texts = read_texts(directory)
    cleaned_texts = preprocess_texts(texts)
    print("Data processing completed.")

    print("Starting embedding computation...")
    # Embedding hesaplama
    embeddings = compute_embeddings(cleaned_texts)
    print("Embedding computation completed.")
    
    print("Original Embeddings Shape:", embeddings.shape)
    print("Original Embeddings Example:", embeddings[0])

    print("Plotting explained variance to determine optimal number of components...")
    # Plot explained variance to determine optimal number of components
    plot_explained_variance(embeddings)

    n_components = int(input("Enter the number of PCA components: "))

    print("Applying PCA...")
    # Boyut indirgeme (PCA kullanarak)
    pca_result = reduce_dimensions(embeddings, n_components=n_components)
    print("PCA completed.")
    
    print("PCA Reduced Embeddings Shape:", pca_result.shape)
    print("PCA Reduced Embeddings Example:", pca_result[0])

    # Compute and print variances
    compute_and_print_variance(embeddings, pca_result)

if __name__ == "__main__":
    main()    