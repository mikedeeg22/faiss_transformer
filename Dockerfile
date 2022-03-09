# base image (may need to modify to the NSF base image)
FROM continuumio/miniconda3

# exposing default port for streamlit
EXPOSE 8081

# making directory of app
WORKDIR /semantic_search

#faiss install
RUN conda install -c conda-forge pytorch faiss-cpu

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install -r requirements.txt

COPY main.py ./main.py
COPY 20newsgroups.csv ./20newsgroups.csv

# cmd to launch app when container is run
CMD streamlit run main.py --server.port 8081