import os
import json
import logging
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import chromadb


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagModel:
    """
    Represents a Retrieval Augmented Generation (RAG) Model.
    """

    def __init__(self, parameters) -> None:
        # Set parameters
        self.llm_model = parameters["llm_model"]
        self.embedding_model = parameters["embedding_model"]
        self.prompt_template = parameters["prompt_template"]
        self.chunk_size = parameters["chunk_size"]
        self.chunk_overlap = parameters["chunk_overlap"]
        self.search_type = parameters["search_type"]
        self.k = parameters["k"]
        self.score_threshold = parameters["score_threshold"]
        self.temperature = parameters["temperature"]
        self.key_username = parameters["key_username"]
        self.key_prompt = parameters["key_prompt"]
        self.files_path = parameters["files_path"]
        self.collection_path = parameters["collection_path"]
        self.collection_name = parameters["collection_name"]

    def data_ingest(self):
        """
        Handles the ingestion of a file from a Flask request and saves it to a specified path.
        Returns: A JSON response from the ingestion process indicating the filename and whether the upload was successful.
        """
        try:
            # Ingest source file into path_files
            logger.info("Capturing the file")
            file = request.files["file"]
            file_path = os.path.join(self.files_path, file.filename)
            file.save(file_path)
            logger.info("File upload completed successfully.")

            response_ingest = {
                "status": "File upload completed successfully.",
                "filename": file.filename
            }
            return jsonify(response_ingest), 200
        except Exception as e:
            logger.error(f"Error during file upload: {e}")
            return jsonify({"error": str(e)}), 500

    def get_request(self, key):
        """
        Retrieves a specific key from a JSON request.
        Returns: The value associated with the key from JSON request.
        """
        return request.json.get(key)

    def generate_chunks(self):
        """
        Processes a PDF file by splitting it into text chunks.
        Returns: A list of fragmented document chunks.
        """
        try:
            file_path = os.path.join(self.files_path, os.listdir(self.files_path)[0])
            logger.info(f"Processing file: {file_path}")

            # TextSplitter configuration
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )

            # Load and split source file (pdf)
            loader = PDFPlumberLoader(file_path)
            docs = loader.load_and_split()

            # Generate chunks from source file (pdf) with TextSplitter instance 
            chunks = text_splitter.split_documents(docs)
            return chunks
        except Exception as e:
            logger.error(f"Error generating chunks: {e}")
            return jsonify({"error": str(e)}), 500

    def vector_embedding(self, chunks, embedding):
        """
        Generates a vector store for document chunks using embeddings.
        Returns: A Chroma vector store instance.
        """
        try:
            logger.info("Creating a vector store...")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                ids=[f"{item.metadata['source']}-{index}" for index, item in enumerate(chunks)],
                collection_name=self.collection_name,
                persist_directory=self.collection_path
            )
            vector_store.persist()
            logger.info("Vector store created successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return jsonify({"error": str(e)}), 500

    def vector_store(self, embedding, username, prompt):
        """
        Stores the prompt and username in the vector store.
        Returns: The updated collection.
        """
        try:
            logger.info("Creating or geting a collection...")
            client = chromadb.PersistentClient(self.collection_path)
            collection = client.get_or_create_collection(name=self.collection_name, embedding_function=embedding)
            collection.add(
                documents=[prompt],
                metadatas=[{"user": username}],
                ids=[hash(prompt)]
            )
            return collection
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
            return jsonify({"error": str(e)}), 500

    def retrieval_chain(self, vector_embedding, llm_model, prompt_template):
        """
        Creates a retrieval chain to process prompts using the RAG Model.
        Returns: A retrieval chain configured with the retriever and document chain.
        """
        try:
            # Generate retrieval instance with vector_embedding
            retriever = vector_embedding.as_retriever(
                search_type=self.search_type,
                #search_kwargs={ "k":self.k, "score_threshold": self.score_threshold }               
                )

            # Generate retrieval chain
            document_chain = create_stuff_documents_chain(llm_model, prompt_template)
            logger.info("Creating retrieval chain...")
            chain = create_retrieval_chain(retriever, document_chain)
            return chain
        except Exception as e:
            logger.error(f"Error creating retrieval chain: {e}")
            return jsonify({"error": str(e)}), 500

    def model_response(self):
        """
        Processes a request and generates a response from the RAG Model.
        Returns: A JSON with the model response including the answer and source information.
        """
        try:
            logger.info("Setting OpenAI key secret...")
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            logger.info("Initializing the LLM model and embedding...")
            if self.llm_model.startswith("gpt-"):
                llm_model = ChatOpenAI(temperature=self.temperature, model=self.llm_model)
                embedding = OpenAIEmbeddings(model=self.embedding_model)
            elif self.llm_model.startswith("llama"):
                llm_model = Ollama(model=self.llm_model)
                embedding = FastEmbedEmbeddings()
            else:
                raise ValueError(f"Unsupported LLM or embedding model: {self.llm_model}")
            
            logger.info("Setting prompt template...")
            prompt_template = PromptTemplate.from_template(self.prompt_template)

            logger.info("Capturing username and prompt from user...")
            username = self.get_request(self.key_username)
            prompt = self.get_request(self.key_prompt)

            #logger.info("Save username and prompt on vectorstore (Chromadb)")
            #vector_store = self.vector_store(embedding, self.key_username, self.key_prompt)

            logger.info("Generating chunks from file source (pdf)...")
            chunks = self.generate_chunks()

            logger.info("Generating vector store...")
            vector_embedding = self.vector_embedding(chunks, embedding)

            logger.info("Generating retrieval chain...")
            chain = self.retrieval_chain(vector_embedding, llm_model, prompt_template)

            logger.info("Generating RAG Model response...")
            response = chain.invoke({"input": prompt})

            logger.info("Generating sources (page_contents) used in the model response...")
            sources = []
            for doc in response["context"]:
                sources.append(
                    {"page_content": doc.page_content}
            )

            logger.info("Setting RAG Model response...")
            model_response = {"Answer": response["answer"], "Sources": sources}
            #model_response = json.dumps({"Answer": response["answer"]})
            return jsonify(model_response), 200
        except Exception as e:
            logger.error(f"Error generating model response: {e}")
            return jsonify({"error": str(e)}), 500