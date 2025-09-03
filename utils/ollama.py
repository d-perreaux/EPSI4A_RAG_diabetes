import ollama
import os

import streamlit as st

import utils.logs as logs

# This is not used but required by llama-index and must be imported FIRST
os.environ["OPENAI_API_KEY"] = "sk-abc123"

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine

###################################
#
# Create Client
#
###################################


def create_client(host: str):
    """
    Creates a client for interacting with the Ollama API.

    Parameters:
        - host (str): The hostname or IP address of the Ollama server.

    Returns:
        - An instance of the Ollama client.

    Raises:
        - Exception: If there is an error creating the client.

    Notes:
        This function creates a client for interacting with the Ollama API using the `ollama` library. It takes a single parameter, `host`, which should be the hostname or IP address of the Ollama server. The function returns an instance of the Ollama client, or raises an exception if there is an error creating the client.
    """
    try:
        client = ollama.Client(host=host)
        logs.log.info("Ollama chat client created successfully")
        return client
    except Exception as err:
        logs.log.error(f"Failed to create Ollama client: {err}")
        return False


###################################
#
# Get Models
#
###################################


def get_models():
    """
    Retrieves a list of available language models from the Ollama server.

    Returns:
        - models (list[str]): A list of available language model names.

    Raises:
        - Exception: If there is an error retrieving the list of models.

    Notes:
        This function retrieves a list of available language models from the Ollama server using the `ollama` library. It takes no parameters and returns a list of available language model names.

        The function raises an exception if there is an error retrieving the list of models.

    Side Effects:
        - st.session_state["ollama_models"] is set to the list of available language models.
    """
    try:
        chat_client = create_client(st.session_state["ollama_endpoint"])
        data = chat_client.list()
        models = []
        for model in data["models"]:
            models.append(model["name"])

        st.session_state["ollama_models"] = models

        if len(models) > 0:
            logs.log.info("Ollama models loaded successfully")
        else:
            logs.log.warn(
                "Ollama did not return any models. Make sure to download some!"
            )

        return models
    except Exception as err:
        logs.log.error(f"Failed to retrieve Ollama model list: {err}")
        return []


###################################
#
# Create Ollama LLM instance
#
###################################


@st.cache_data(show_spinner=False)
def create_ollama_llm(model: str, base_url: str, system_prompt: str = None, request_timeout: int = 60) -> Ollama:
    """
    Create an instance of the Ollama language model.

    Parameters:
        - model (str): The name of the model to use for language processing.
        - base_url (str): The base URL for making API requests.
        - request_timeout (int, optional): The timeout for API requests in seconds. Defaults to 60.

    Returns:
        - llm: An instance of the Ollama language model with the specified configuration.
    """
    try:
        # Settings.llm = Ollama(model=model, base_url=base_url, system_prompt=system_prompt, request_timeout=request_timeout)
        Settings.llm = Ollama(model=model, base_url=base_url, request_timeout=request_timeout)
        logs.log.info("Ollama LLM instance created successfully")
        return Settings.llm
    except Exception as e:
        logs.log.error(f"Error creating Ollama language model: {e}")
        return None


###################################
#
# Chat (no context)
#
###################################


def chat(prompt: str):
    """
    Initiates a chat with the Ollama language model using the provided parameters.

    Parameters:
        - prompt (str): The starting prompt for the conversation.

    Yields:
        - str: Successive chunks of conversation from the Ollama model.
    """

    try:
        llm = create_ollama_llm(
            st.session_state["selected_model"],
            st.session_state["ollama_endpoint"],
        )
        stream = llm.stream_complete(prompt)
        for chunk in stream:
            yield chunk.delta
    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        return


###################################
#
# Document Chat (with context)
#
###################################


def context_chat(prompt: str, query_engine: RetrieverQueryEngine):
    """
    Initiates a chat with context using the Llama-Index query_engine.

    Parameters:
        - prompt (str): The starting prompt for the conversation.
        - query_engine (RetrieverQueryEngine): The Llama-Index query engine to use for retrieving answers.

    Yields:
        - str: Successive chunks of conversation from the Llama-Index model with context.

    Raises:
        - Exception: If there is an error retrieving answers from the Llama-Index model.

    Notes:
        This function initiates a chat with context using the Llama-Index language model and index.

        It takes two parameters, `prompt` and `query_engine`, which should be the starting prompt for the conversation and the Llama-Index query engine to use for retrieving answers, respectively.

        The function returns an iterable yielding successive chunks of conversation from the Llama-Index index with context.

        If there is an error retrieving answers from the Llama-Index instance, the function raises an exception.

    Side Effects:
        - The chat conversation is generated and returned as successive chunks of text.
    """

    try:
        streaming_response = query_engine.query(prompt)
        response_generator = streaming_response.response_gen
        initial_buffer = ""
        is_thought_processed = False

        for text_chunk in response_generator:
            if not is_thought_processed:
                initial_buffer += text_chunk
                end_tag = "</think>"
                end_tag_pos = initial_buffer.find(end_tag)
                if end_tag_pos != -1:
                    # On récupère le vrai début de la réponse (ce qui est APRÈS la balise)
                    true_response_start = initial_buffer[end_tag_pos + len(end_tag):]
                    
                    # On envoie ce premier morceau de la vraie réponse
                    yield true_response_start
                    
                    # On marque le traitement comme terminé
                    is_thought_processed = True
            else:
                # Une fois le bloc <think> passé, on streame normalement
                yield text_chunk

        # Après la génération de la réponse, on gère l'affichage des sources
        source_nodes = streaming_response.source_nodes
        
        if source_nodes:
            unique_sources = set()
            for node in source_nodes:
                if 'file_name' in node.node.metadata:
                    unique_sources.add(node.node.metadata['file_name'])

            if unique_sources:
                sources_text = "\n\n---\n**Sources utilisées :**\n"
                for source in sorted(list(unique_sources)):
                    sources_text += f"- `{source}`\n"
                
                yield sources_text

    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        return
