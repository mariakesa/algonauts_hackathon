## Hackathon with Algonauts 2023 Challenge data

### Getting started
1. Install poetry for dependence management
2. Clone this repository
3. Inside the cloned repository, install deps with `poetry install`
4. Download Algonauts data and extract it.
5. `export DATA=my/data/path` to point to the extracted data directory.
The directory should contain subdirectories for each subject, like `subj01/`
6. `export DATA=my/db/path` to a directory where a Milvus database can be created and stored
7. `poetry run python get_vectors.py` to extract image embeddings and store them to database