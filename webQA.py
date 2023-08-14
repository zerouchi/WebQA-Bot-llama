from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

import streamlit as st


llm = Ollama(base_url="http://localhost:11434", 
             model="llama2", 
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

def load_data(url,query):
    
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
    
    
    
    
    docs = vectorstore.similarity_search(query)
    
    template = """ firstly greet the user if user says hii does not use your knowledge .
    then the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever= vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    
    result = qa_chain({"query": query})  
    
    return result["result"] 

def main():
    st.title("WebQA Bot")
    
    
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""

    # Sidebar section
    st.sidebar.header("Enter your URL")
    url_input = st.sidebar.text_input("URL", st.session_state.input_value)

    # Store input_value in session_state
    st.session_state.input_value = url_input
    

# Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    respo = ""        
    # React to user input
    if prompt := st.chat_input("Write your query?"):
        respo = load_data(url_input,prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
    
    response = f"{respo}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if response == "":
            st.markdown("Hello there! 👋 I'm the WebQA bot, designed to assist you in extracting information from websites and providing answers to your queries based on the content of the given URL. Just provide me with the URL of a website you'd like me to analyze, and I'll do my best to extract relevant information and answer any questions you have about the content on that website. Feel free to ask me anything related to the website's content, and I'll do my utmost to provide you with accurate and helpful responses. Let's dive into the world of web-based knowledge together!")
        else:    
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()


    
    