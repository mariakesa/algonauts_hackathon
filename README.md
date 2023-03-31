## Hackathon with Algonauts 2023 Challenge data

### Getting started
1. Install poetry for dependence management
2. Clone this repository
3. Inside the cloned repository, install deps with `poetry install`
4. Download Algonauts data and extract it.
5. `export DATA=my/data/path` to point to the extracted data directory.
The directory should contain subdirectories for each subject, like `subj01/`
6. `poetry run milvus-server --data db` in another terminal to run Milvus and use the `./db` directory for its storage and config.
7. `poetry run python get_vectors.py` to extract image embeddings and store them to database