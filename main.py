from util import *

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ''

# Configure the settings of the webpage
st.set_page_config(page_title="Chat with Websites", page_icon="🧊", layout="wide")

# Add a header
st.header("WebInquisitor 💬🤖")
st.write(":blue[***Explore Websites with AI -- Chat with Multiple Websites***]")
st.write("*WebInquisitor* is a cutting-edge application designed to transform how you interact with websites. "
         "Simply input a website URL(s), and it will leverage advanced artificial intelligence to answer any question "
         "you have related to provided website(s). ")

# Sidebar Configuration
st.sidebar.header("Configurations")
app_unlock = configure_apikey_sidebar()
urls = configure_sidebar_url(app_unlock)

process = st.sidebar.button("Process URLs", type='primary', disabled=not urls)
if process:
    with st.spinner("Processing..."):
        docs = data_ingestion(urls=urls)  # Ingest data
        st.session_state.embeddings = get_embeddings()  # Google Generative AI embeddings
        build_vector_store_database(documents=docs,
                                    embeddings=st.session_state.embeddings)  # Create vector store database
        st.sidebar.success("Done")

configure_about_sidebar()

# Input question from the user
st.subheader('Enter your Question:')
user_question = st.text_input("Ask a question from the Website", disabled=not urls, label_visibility='collapsed')
submit = st.button('Submit', type='primary', disabled=not user_question)
if submit:
    with st.spinner("Thinking"):
        # Load data
        faiss_index = FAISS.load_local("faiss_index", st.session_state.embeddings,
                                       allow_dangerous_deserialization=True)
        # Load LLM
        llm = get_llm(st.session_state.api_key)
        st.write(get_response_llm(llm, faiss_index, user_question))
        st.success("Done")
