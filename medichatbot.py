import os
import streamlit as st

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Define the path to the FAISS vector store
DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """
    Loads and caches the FAISS vector store.
    This function uses st.cache_resource to avoid reloading the vector store
    on every Streamlit rerun.
    """
    try:
        embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        # Load the local FAISS vector store
        db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS database: {e}")
        st.error("Please ensure the 'vectorstore/db_faiss' directory exists and contains valid FAISS index files.")
        return None


def set_custom_prompt(custom_prompt_template):
    """
    Sets up a custom prompt template for the QA chain.

    Args:
        custom_prompt_template (str): The template string for the prompt.

    Returns:
        PromptTemplate: A Langchain PromptTemplate instance.
    """
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    """
    Loads a HuggingFace Endpoint LLM.

    Args:
        huggingface_repo_id (str): The ID of the HuggingFace model to use.
        HF_TOKEN (str): The HuggingFace API token for authentication.

    Returns:
        HuggingFaceEndpoint: An instance of the HuggingFaceEndpoint LLM.
    """
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        # Pass the token directly as huggingfacehub_api_token for authentication
        huggingfacehub_api_token=HF_TOKEN,
        # Pass max_new_tokens directly, not inside model_kwargs, to control output length
        max_new_tokens=512
    )
    return llm


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Ask Medi Chatbot!")

    # Initialize chat history in session state if not already present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Get user input from the chat prompt
    prompt=st.chat_input("Pass your prompt here")


    if prompt:
        # Display user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        # Define the custom prompt template for the RAG chain
        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer.
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        # Model ID changed to one that supports text-generation through the HuggingFaceEndpoint
        HUGGINGFACE_REPO_ID="HuggingFaceH4/zephyr-7b-beta"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        # Ensure HF_TOKEN is available
        if not HF_TOKEN:
            st.error("HuggingFace API token (HF_TOKEN) not found. Please set it in your .env file.")
            st.session_state.messages.append({'role':'assistant', 'content': "Error: HuggingFace API token not found."})
            return

        try:
            # Load the vector store (cached)
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.session_state.messages.append({'role':'assistant', 'content': "Error: Could not load vector store."})
                return

            # Initialize the LLM and the QA chain
            llm_instance = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
            qa_chain=RetrievalQA.from_chain_type(
                llm=llm_instance,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Invoke the QA chain with the user's prompt
            response=qa_chain.invoke({'query':prompt})

            # Extract result and source documents
            result=response["result"]
            source_documents=response["source_documents"]

            # Format the response to show result and sources, including page number if available
            result_to_show=result+"\n\n**Source Docs:**\n"
            for doc in source_documents:
                # Assuming 'page_content' holds the text and 'source' holds metadata
                source_info = doc.metadata.get('source', 'N/A')
                page_info = doc.metadata.get('page', 'N/A') # Get the page number from metadata
                result_to_show += f"- {doc.page_content[:150]}... (Source: {source_info}, Page: {page_info})\n"


            # Display assistant's message
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            # Display error message in the chat
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({'role':'assistant', 'content': f"An error occurred: {str(e)}"})

# Run the Streamlit app
if __name__ == "__main__":
    main()
