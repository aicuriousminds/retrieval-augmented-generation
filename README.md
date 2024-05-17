## Contents
[RAG Model](#rag-model-robot)<br>
- [Overview](#mag-overview)<br>
- [Components](#open_file_folder-components)<br>
- [Tools Used](#hammer_and_wrench-tools-used)<br>
- [Getting Started](#rocket-getting-started)<br>
<!--- - [References](#books-references)<br> -->

# RAG Model :robot:

**`Retrieval Augmented Generation (RAG) with GPT-4o.`**

## :mag: Overview
Recently, OpenAI launched the GPT-4o model, claimed to be the world's most powerful according to benchmarks. This project evaluates its document search capabilities within a simple Retrieval Augmented Generation (RAG) Model. The application, built with Flask, uses the GPT-4o model to provide answers based on specific documents.

## :open_file_folder: Components
Below is a breakdown of the key components included in this repository:

- [**`docs/`**](docs/readme.md): Project documentation.
- [**`tests/`**](tests/readme.md): Test cases documentation.
- [**`src/`**](src/): Source files for the application.
  - [`__init__.py`](src/__init__.py): Initializes src as a Python module.
  - [`main.py`](src/main.py): Main script to run the application.
  - [`app.py`](src/app.py): Flask application endpoints.
  - [`rag_model.py`](src/rag_model.py): RAG Model functionality and methods.
- [**`config/`**](config/): Configuration files.
  - [`parameters.json`](config/parameters.json): Configuration settings for RAG Model.
- [**`scripts/`**](scripts/): Additional scripts for utilities.
  - [`fine_tuning.py`](scripts/fine_tuning.py): In progress.
- [**`.env`**](.env): Set openai_api_key.
- [**`.gitignore`**](.gitignore): Specifies intentionally untracked files to ignore.
- [**`requirements.txt`**](requirements.txt): Python dependencies.
- [**`README.md`**](README.md): Detailed description of the project.
- [**`LICENSE`**](LICENSE): MIT License information.


## :hammer_and_wrench: Tools Used
The following tools are utilized in this project:

1. **LangChain**
2. **Flask**
3. **GPT-4o**
4. **ChromaDB**

## :rocket: Getting Started
Follow these steps to set up and run the project on your local machine:

1. **Clone the repository:**

``` bash
git clone https://github.com/jairzinhosantos/rag-model.git
```

3. **Set up a virtual environment:**

``` bash
python -m venv venv
source venv/bin/activate
```

4. **Install dependencies:**

``` bash
pip install -r requirements.txt
```

5. **Run the application:**

``` bash
python src/main.py
```


<!---
## :books: References
[^1]: [x](y)
[^2]: [x](y)
[^3]: [x](y)
[^4]: [x](y)
-->

