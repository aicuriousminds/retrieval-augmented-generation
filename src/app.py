from flask import Flask, request, jsonify, make_response
from rag_llm import RagLlm
import json
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration parameters from JSON file
def load_parameters():
    parameters_path = "config/parameters.json"
    try:
        with open(parameters_path) as file:
            parameters_list = json.load(file)
        return parameters_list
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {parameters_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the configuration file at {parameters_path}")
        return {}

# Initialize RAG LLM with loaded parameters
parameters = load_parameters()
rag_llm = RagLlm(parameters)

@app.route("/data_ingestion", methods=["POST"])
def data_ingestion():
    """
    API endpoint to ingest data. The method is set to POST, meaning it expects data to be sent to this endpoint for processing.
    Invokes the data_ingest method to perform data ingestion.
    """
    try:
        logger.info("Running data_ingest instance...")
        response_ingest = rag_llm.data_ingest()
        return response_ingest
        #return jsonify(response_ingest), 200
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/rag_llm", methods=["POST"])
def rag_llm_endpoint():
    """
    API endpoint to process requests using the RAG LLM model.
    Invokes the model_response method to generate RAG LLM model response.
    """
    try:
        logger.info("Running model_response instance...")
        model_response = rag_llm.model_response()
        return model_response
        #return make_response(jsonify(model_response), 200)
    except Exception as e:
        logger.error(f"Error during RAG LLM processing: {e}")
        return jsonify({"error": str(e)}), 500

def start_app():
    """
    Starts the Flask application.
    """
    # Set local parameters
    host=parameters["host"]
    port=parameters["port"]

    # Run app
    app.run(host=host, port=port, debug=True)

# if __name__ == "__main__":
#      start_app()