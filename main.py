import pandas as pd
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer


@st.cache
def read_data(data_loc="20newsgroups.csv"):
    """Read the data from directory"""
    raw = pd.read_csv(data_loc)
    sample = raw.sample(n=10000)
    return sample


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(suppress_st_warning=True)
def make_embeddings(model, df):
    return model.encode(df.content.to_list(), show_progress_bar=True)


@st.cache(suppress_st_warning=True)
def faiss_steps(data, embeddings):
    # Step 1: Change Data type
    embed_processed = np.array([embedding for embedding in embeddings]).astype('float32')
    # Step 2: Instantiate the index
    index = faiss.IndexFlatL2(embed_processed.shape[1])
    # Step 3: Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)
    # Step 4: Add vectors and their IDs
    index.add_with_ids(embed_processed, data.id.values)
    return index


def main():
    # load data and models
    st.title('20 News Group Dataset Semantic Search')
    df = read_data()
    model = load_bert_model()
    embeddings = make_embeddings(model, df)
    faiss_index = faiss_steps(df, embeddings)
    # streamlit settings
    st.subheader('Enter terms in search box and press ctrl+enter to begin')
    user_input = st.text_area("Search box")
    search_n = st.number_input('Enter the number of search results to return',
                               min_value=2, max_value=100, value=10, step=1)
    # fetch results
    if user_input:
        # get vector from query
        query_vec = model.encode(user_input)
        query_vec = np.array([query_vec]).astype('float32')
        # get top 10 closest documents
        D, I = faiss_index.search(query_vec, k=search_n)
        title_list = [list(df[df.id == idx]['title']) for idx in I[0]]
        # create dataframe for display
        results_df = pd.DataFrame({'id': I[0], 'title': title_list, 'L2_Distance': D[0]})
        st.dataframe(results_df)
        st.download_button(label='Download Search Results as CSV', data=results_df.to_csv(),
                           file_name='Search_results.csv')


if __name__ == "__main__":
    main()
