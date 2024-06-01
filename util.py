from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import streamlit as st



# Function to configure sidebar to verify and get the model  api key
def configure_apikey_sidebar():
    st.session_state.api_key = st.sidebar.text_input(f'Enter Google API Key', type='password',
                                                     help='Get Google API Key from: https://aistudio.google.com/app/apikey')
    if st.session_state.api_key == '':
        st.sidebar.warning('Enter the API key')
        app_unlock = False

    elif st.session_state.api_key.startswith('AI') and (len(st.session_state.api_key) == 39):
        st.sidebar.success(' Proceed!', icon='️👉')
        app_unlock = True
    else:
        st.sidebar.warning('Please enter the correct credentials!', icon='⚠️')
        app_unlock = False

    return app_unlock


def configure_sidebar_url(app_unlock):
    st.sidebar.subheader("Enter Website URLs:")
    urls = []  # List of urls
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i + 1}", disabled=not app_unlock)
        # Append URLs if provided
        if url:
            urls.append(url)

    return urls


def configure_about_sidebar():
    st.sidebar.subheader("About")
    with st.sidebar.expander('How to use this App'):
        st.markdown(''' 
            - Unlock the app by entering your Google API Key. Get your API key from: 
            https://aistudio.google.com/app/apikey
            - Enter Website URL(s) and click on Process URLs button
            - Enter your question related to provided websites
            ''')
    with st.sidebar.expander('Source Code'):
        st.markdown(''' Source code: [GitHub](https://github.com/mzeeshanaltaf/chat-with-multiple-websites)''')
    with st.sidebar.expander('Contact'):
        st.markdown(''' Any Queries: Contact [Zeeshan Altaf](mailto:zeeshan.altaf@gmail.com)''')
    with st.sidebar.expander('Technology'):
        st.markdown(''' 
            - LLM: Google Gemini - gemini-1.5-pro-latest
            - Embeddings: Google Text Embeddings - text-embedding-004
            - Vector Store -- FAISS''')


def data_ingestion(urls):
    loader = WebBaseLoader(urls)
    documents = loader.load()  # Load html
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)  # Text splitter
    chunks = text_splitter.split_documents(documents)  # Split data into chunks
    return chunks


def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings


def build_vector_store_database(documents, embeddings):
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Location where the vector store database is saved


def get_llm(api_key):
    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=api_key)
    return llm


def get_response_llm(llm, vector_store, query):
    prompt_template = """
    Human: Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
        Don't exceed 250 words on the explanation. If you don't know the answer, just say that you don't know, 
        don't try to make up an answer.\n

        Context:\n {context}?\n
        Question: {question}

        Assistant:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa.invoke({"query": query})
    return answer['result']
