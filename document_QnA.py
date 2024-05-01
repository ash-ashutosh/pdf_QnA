import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
    elif file_extension.lower() in [".doc", ".docx"]:
        loader = TextLoader(file_path)
        pages = loader.load_and_split()
    else:
        st.error("Unsupported file format. Please upload a PDF or Word document.")
        return None

    return pages

def main():
    st.title("Document Search and Question Answering App")

    # Retrieve OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        return

    # File upload UI
    st.subheader("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "doc", "docx"])

    if uploaded_file:
        # Save the uploaded file to a temporary location
        with open("temp_file" + os.path.splitext(uploaded_file.name)[1], "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and process the document
        pages = load_document("temp_file" + os.path.splitext(uploaded_file.name)[1])

        if pages:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(pages)

            # Generate embeddings for text chunks
            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = FAISS.from_documents(
                documents=texts,
                embedding=embedding
            )

            # Load question answering model
            chain = load_qa_chain(
                OpenAI(openai_api_key=openai_api_key),
                chain_type="stuff"
            )

            # Streamlit UI for question input
            query = st.text_input("Enter your question:")
            if st.button("Search"):
                answer = vectordb.similarity_search(query)
                if answer:
                    result = chain.run(input_documents=answer, question=query)
                    st.write("Answer:", result)
                else:
                    st.write("No relevant answer found.")

        # Clean up temporary file
        os.remove("temp_file" + os.path.splitext(uploaded_file.name)[1])

if __name__ == "__main__":
    main()
