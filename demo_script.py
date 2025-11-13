import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# Import the correct classes for Google GenAI within LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# Import the chain functions correctly from the new LangChain structure
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


# >>>>>>>>> ADD YOUR API KEY HERE <<<<<<<<<
# It is best practice to set the API key as an environment variable
# The LangChain classes will automatically detect the GOOGLE_API_KEY environment variable.
os.environ["GOOGLE_API_KEY"] = "AIzaSyAj4P1NsF-R6-Gl9SkDzvnlBvr-3SPyTCM"

# Note: The 'google-genai' SDK specific calls that were conflicting have been removed.

# --- 1. Load Data (Make sure you have a file in a 'data' folder) ---
# Example for a text file:
# Make sure to create a 'data' folder and add a 'sample_doc.txt' file
loader = TextLoader("./data/sample_doc.txt")
docs = loader.load()

# --- 2. Split Data into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# --- 3. Create Embeddings & Vector Store (FAISS runs locally) ---
# Initialize the Google GenAI Embeddings class
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

# --- 4. Define the LLM (Using Google's model within LangChain) ---
# Initialize the ChatGoogleGenerativeAI class with model and parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# --- 5. Create RAG Chain ---
# This defines the prompt to use with the retrieved context
prompt = ChatPromptTemplate.from_template("""
Answer the user's questions based ONLY on the provided context.
If the answer is not found in the context, clearly state that you don't know the answer.

Context: {context}

Question: {input}
""")

# The 'llm' object from ChatGoogleGenerativeAI is passed here
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- 6. Live Demo Function ---
def ask_question_live(question):
    print(f"\n--- Asking: {question} ---")
    # Use the invoke method to run the chain
    response = retrieval_chain.invoke({"input": question})
    print(f"--- AI Answer: ---\n{response['answer']}")
    print("-------------------\n")


# --- 7. Run the Demo (In your presentation, change the questions live) ---
if __name__ == "__main__":
    # Change 'your_document_name.pdf' above to your file name
    print("Wlecome to Abhi's RAG Demo. Type 'Exit' to quit.")
    # while True:
    # user_question = input("\n Enter your question: ")
    # if user_question.lower() == 'Exit':
    # print("Existing RAG system demo.")
    # break

    # ask_question_live(user_question)


st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
