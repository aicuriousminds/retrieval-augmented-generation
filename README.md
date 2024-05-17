## Contents
[Challenge RAG with LLMs](#challenge-rag-with-llms-robot)<br>
- [Challegence Overview](#mag-challenge-overview)<br>
- [Files and Directories](#open_file_folder-files-and-directories)<br>

[Solution to Challenge RAG with LLMs](#solution-to-challenge-rag-with-llms-brainbulb)<br>
- [Solution Overview](#mag-solution-overview)<br>
- [Components](#open_file_folder-components)<br>
- [Tools Used](#hammer_and_wrench-tools-used)<br>
- [Getting Started](#rocket-getting-started)<br>
<!--- - [References](#books-references)<br> -->

---

<a href='https://piconsulting.com.ar/'>
    <img src='assets/look-and-feel/icons/pi-dsc/pi-data-strategy-consulting.png' alt='icon | pi-dsc' width='500px'/></a>

# Challenge RAG with LLMs :robot:

**`This repository has been developed as part of the submission for the AI Engineer role selection process at PI Data Strategy & Consulting.`**

## :mag: Challenge Overview
This repository contains all files and documentation pertaining to submission for the Challenge RAG with LLMs. Each component of the challenge is outlined below, complete with links to detailed descriptions and resources used.

## :open_file_folder: Files and Directories
Below is a table summarizing the main components of project. It includes detailed task descriptions and implementation of the solution:

| Contributor | Document | Description | Link |
| --- | --- | --- | --- |
| `PI Data Strategy & Consulting` | `Challenge RAG with LLMs` | Detailed challenge description. | [more...](assets/challenge/readme.md) |
| `Jairzinho Santos` | `Solution to Challenge RAG with LLMs` | Implementation and solution. | [more...](#solution-to-challenge-rag-with-llms-brainbulb) |

<br>

> [!NOTE]
> This README is designed to facilitate the review of submission for the Challenge RAG with LLMs. For further details or updates, feel free to revisit this page or contact me directly.

<br>
<br>

# Solution to Challenge RAG with LLMs :brain::bulb:

This repository contains the implementation of the " Solution Challenge RAG with LLMs", developed as part of the AI Engineer role selection process at PI Strategy & Consulting.

## :mag: Solution Overview
This solution focuses on creating a simplified RAG (Retrieved Augmented Generation) system using a Python-based API that interacts with a Large Language Model (LLM) to provide specific document-based responses to user queries.

## :open_file_folder: Components
Here is a breakdown of the key components in this repository:

- [**`docs/`**](docs/readme.md): Project documentation.

- [**`tests/`**](tests/readme.md): Test cases for the project.

- [**`src/`**](src/): Source files for the application.
  - [`__init__.py`](src/__init__.py): Makes src a Python module.
  - [`app.py`](src/app.py): Flask application endpoint.
  - [`rag_llm.py`](src/rag_llm.py): RAG LLM Model functionality and methods.

- [**`config/`**](config/): Configuration files.
  - [`parameters.json`](config/parameters.json): Configuration settings for RAG LLM Model.

- [**`scripts/`**](scripts/): Additional scripts for utilities.
  - [`fine_tuning.py`](scripts/fine_tuning.py): Script for fine-tuning the model.

- [**`Dockerfile`**](Dockerfile): Docker configuration for containerization.

- [**`requirements.txt`**](requirements.txt): Python dependencies.

- [**`.gitignore`**](.gitignore): Specifies intentionally untracked files to ignore.

- [**`README.md`**](README.md): Detailed description of the project.

- [**`LICENSE`**](LICENSE): MIT License information.


## :hammer_and_wrench: Tools Used
Below are the tools used in the project.
1. **LangChain**
2. **Flask**
3. **GPT-4o**
4. **GitHub**
5. **ChromaDB**


## :rocket: Getting Started
Follow these steps to get the project up and running on your local machine:

1. **Clone the repository:** <br>
`git clone https://github.com/jairzinhosantos/challenge-rag-llm.git`

3. **Set up a virtual environment:** <br>
`python -m venv venv` <br>
`source venv/bin/activate`

4. **Install dependencies:** <br>
`pip install -r requirements.txt`

5. **Run the application:** <br>
`python src/app.py`


<!---
## :books: References
[^1]: [x](y)
[^2]: [x](y)
[^3]: [x](y)
[^4]: [x](y)
-->
