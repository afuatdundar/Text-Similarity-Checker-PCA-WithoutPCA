import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from clustering import reduce_dimensions
from data_processing import preprocess_texts, read_texts
from embedding import compute_embeddings
import matplotlib.pyplot as plt
import numpy as np
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

def compute_and_save_similarity_matrices(embeddings, pca_result, texts, altered_embeddings, altered_pca_result, altered_texts, output_file):
    # Compute cosine similarities for embeddings
    cos_sim_matrix_embeddings = cosine_similarity(embeddings)
    # Compute cosine similarities for PCA results
    cos_sim_matrix_pca = cosine_similarity(pca_result)
    
    # Compute cosine similarities for embeddings and PCA results with altered texts
    cos_sim_matrix_embeddings_altered = cosine_similarity(embeddings, altered_embeddings)
    cos_sim_matrix_pca_altered = cosine_similarity(pca_result, altered_pca_result)
    
    # Save cosine similarity matrices to Excel
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        pd.DataFrame(cos_sim_matrix_embeddings, index=texts, columns=texts).to_excel(writer, sheet_name='Sim Matrix (Embeddings)')
        pd.DataFrame(cos_sim_matrix_pca, index=texts, columns=texts).to_excel(writer, sheet_name='Sim Matrix (PCA)')
        pd.DataFrame(cos_sim_matrix_embeddings_altered, index=texts, columns=altered_texts).to_excel(writer, sheet_name='Sim Matrix (Embeddings vs Alt)')
        pd.DataFrame(cos_sim_matrix_pca_altered, index=texts, columns=altered_texts).to_excel(writer, sheet_name='Sim Matrix (PCA vs Alt)')
    
    print(f"Similarity matrices saved to {output_file}")

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
    altered_pca_result = reduce_dimensions(altered_embeddings, n_components=pca_n_components)
    
    # Compute and save similarity scores
    output_file = 'similarity_scores.xlsx'
    compute_and_save_similarity_matrices(embeddings, pca_result, cleaned_texts, altered_embeddings, altered_pca_result, cleaned_altered_texts, output_file)

if __name__ == "__main__":
    main()
