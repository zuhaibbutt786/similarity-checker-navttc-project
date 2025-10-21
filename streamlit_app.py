import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textract
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os # For path operations
import tempfile # For creating temporary files

# --- NLTK Downloads (Important for initial setup) ---
def download_nltk_data():
    data_packages = ['stopwords', 'punkt','punkt_tab' ,'wordnet', 'omw-1.4']
    for package in data_packages:
        try:
            nltk.data.find(f'corpora/{package}')
        except Exception:
            st.info(f"Downloading NLTK package: {package}...")
            nltk.download(package, quiet=True)
            st.success(f"NLTK package '{package}' downloaded.")

download_nltk_data()

# --- Global NLTK Resources ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Text Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📄 Document Similarity Checker")
st.markdown("Upload multiple PDF, DOCX, or TXT files to find out how similar they are!")

uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Select one or more files from your computer."
)

if uploaded_files:
    documents_content = {}
    st.info(f"Processing {len(uploaded_files)} documents...")
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    
    # --- File Reading and Text Extraction ---
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar_text.text(f"Extracting text from: {uploaded_file.name}...")
        
        # Determine file extension from uploaded_file.name
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        try:
            # textract.process now gets a real file path
            if file_extension == ".pdf":
                text = textract.process(temp_file_path, extension='.pdf', encoding='utf-8').decode('utf-8')
            elif file_extension == ".docx":
                text = textract.process(temp_file_path, extension='.docx', encoding='utf-8').decode('utf-8')
            elif file_extension == ".txt":
                # For plain text, textract can handle it, or you can read directly
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                st.warning(f"Unsupported file type for {uploaded_file.name}. Skipping.")
                text = "" # Set text to empty to avoid errors
                
            if text: # Only add if text was successfully extracted
                documents_content[uploaded_file.name] = text
            
        except Exception as e:
            st.error(f"Could not extract text from **{uploaded_file.name}**: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar_text.empty()
    progress_bar.empty()
    
    if len(documents_content) < 2:
        st.warning("Please upload at least two documents containing meaningful text to compare their similarities.")
    elif documents_content:
        st.success(f"Successfully extracted text from {len(documents_content)} documents.")
        st.info("Now calculating similarities using TF-IDF and Cosine Similarity...")
        
        with st.spinner("Applying NLP preprocessing (tokenization, lemmatization, stop word removal)..."):
            processed_texts = {name: preprocess_text(content) for name, content in documents_content.items()}
        
        filtered_processed_texts = {name: text for name, text in processed_texts.items() if text.strip()}

        if len(filtered_processed_texts) < 2:
            st.error("After preprocessing, less than two documents contain meaningful text for comparison. Please upload documents with more content.")
        else:
            vectorizer = TfidfVectorizer()
            doc_names_for_vectorization = list(filtered_processed_texts.keys())
            tfidf_matrix = vectorizer.fit_transform(list(filtered_processed_texts.values()))
            
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_df = pd.DataFrame(similarity_matrix, index=doc_names_for_vectorization, columns=doc_names_for_vectorization)
            
            st.subheader("📊 Document Similarity Matrix (Cosine Similarity)")
            st.markdown("A score of 1.0 means identical, 0.0 means no common terms.")
            st.dataframe(similarity_df.style.format("{:.3f}"), use_container_width=True)

            st.subheader("🌟 Top Similar Document Pairs")
            st.markdown("Pairs are sorted by their similarity score (highest first).")
            
            pairs = []
            for i in range(len(doc_names_for_vectorization)):
                for j in range(i + 1, len(doc_names_for_vectorization)):
                    pairs.append((doc_names_for_vectorization[i], doc_names_for_vectorization[j], similarity_matrix[i, j]))
            
            pairs_df = pd.DataFrame(pairs, columns=['Document 1', 'Document 2', 'Similarity Score'])
            pairs_df = pairs_df.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)
            
            st.dataframe(pairs_df.style.format({"Similarity Score": "{:.3f}"}), use_container_width=True)

            # --- Similarity Heatmap ---
            st.subheader("🔥 Similarity Heatmap")
            st.markdown("A visual representation of document similarities.")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                similarity_df, 
                annot=True, 
                cmap='viridis',
                fmt=".2f", 
                linewidths=.5, 
                cbar_kws={'label': 'Cosine Similarity'},
                ax=ax
            )
            ax.set_title('Document Similarity Heatmap', fontsize=16)
            
            # --- CORRECTED TICK PARAMETERS ---
            # Set rotation for x-axis tick labels
            ax.tick_params(axis='x', rotation=45) 
            # Set horizontal alignment for x-axis tick labels
            plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor") 
            
            ax.tick_params(axis='y', rotation=0) # Y-axis rotation is fine
            
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("---")
            st.info("💡 **How it works:** Text is extracted, cleaned (lowercase, punctuation removed, stop words removed, words lemmatized), then converted into numerical TF-IDF vectors. Finally, Cosine Similarity is calculated between these vectors.")
