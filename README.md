# faiss_transformer

Uses the distilbert-base-nli-stsb-mean-tokens sentence transformer model to encode the 20 news group dataset's content field.  A FAISS index is created from the data to enable search from a user's input query.  A simple streamlit front-end provides interaction with the model (text area for user input, number of search results returned, download button for results, etc.)
