import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import openai
#from htmlTemplates import css, bot_template, user_template

with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/Nash242/OpenAI_Demo/main/client.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/Nash242/OpenAI_Demo/main/chatbot.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''


def load_docs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(content)
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.create_documents(documents)
    return chunks

def create_vectorstore(persist_directory, text_chunks, user_question):
    embeddings = OpenAIEmbeddings()
    
    # Check if the persistent vector database already exists
    if os.path.exists(persist_directory):
        final_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
        final_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        final_db.similarity_search_with_score(user_question,k=3)
    return final_db

def create_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    if callable(st.session_state.conversation):
        if (user_question.lower().strip() in ['hi','hello','good morning','good afternoon','good evening']):
            question = user_question
        else:
            question = user_question + " Answer in step by step points only"
        response = st.session_state.conversation({'question': question})
        if st.session_state.chat_history is None:
            st.session_state.chat_history = []
        st.session_state.chat_history = st.session_state.chat_history + response['chat_history']
        print(st.session_state.chat_history)
        for i, message in list(enumerate(st.session_state.chat_history)):
            if i % 2 != 0:
                val = ""
                for j in range(len(message.content)):
                    if message.content[j].isdigit() and j < len(message.content)-1 and message.content[j+1] == ".":
                        val += "\n" + message.content[j]
                    else:
                        val += message.content[j]
                message.content = val
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content.split("Answer in step by step points only")[0]), unsafe_allow_html=True)
    else:
        st.error("Conversation chain is not initialized. Please process documents first.")


def handle_user_input2(user_question):
    if callable(st.session_state.conversation):
        if (user_question.lower().strip() in ['hi','hello','good morning','good afternoon','good evening']):
            question = user_question
        else:
            question = user_question + " Answer in step by step points only"
        response = st.session_state.conversation({'question': question})
        if st.session_state.chat_history is None:
            st.session_state.chat_history = []
        st.session_state.chat_history = response['chat_history'] + st.session_state.chat_history
        print(st.session_state.chat_history)
        for i, message in list(enumerate(st.session_state.chat_history)):
            if i % 2 != 0:
                val = ""
                for j in range(len(message.content)):
                    if message.content[j].isdigit() and j < len(message.content)-1 and message.content[j+1] == ".":
                        val += "\n" + message.content[j]
                    else:
                        val += message.content[j]
                message.content = val
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content.split("Answer in step by step points only")[0]), unsafe_allow_html=True)
    else:
        st.error("Conversation chain is not initialized. Please process documents first.")

def disable(b):
    st.session_state["disabled"] = b


def main():
    # os.environ['AZURE_OPENAI_API_KEY'] = ""
    # os.environ['AZURE_OPENAI_ENDPOINT'] = ""
    # openai.api_type = ""
    # openai.api_key = ""
    # openai.api_base = ""
    # openai.api_version = ""
    
    os.environ["OPENAI_API_KEY"] = "OpenAI_Key"

#     st.set_page_config(page_title="Chat with multiple Files",
#                        page_icon=":books:",
# )
    #st.markdown(pg_bg_img, unsafe_allow_html=True)
    #st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "disabled" not in st.session_state:
        st.session_state["disabled"] = False

    columns = st.columns(2)

    columns[0].image("bot4.png")


    # with col1:
    #     st.image("bot4.png")

    # with col2:
    #     st.header("GenX")

    # st.image("bot4.png")
    st.header("Chat with multiple files :books:")

    st.write("Following are the examples, when clicked will show How to chat with your data")
  
    if st.button("Request to access the Dashboard?", key='demo1',on_click=disable, args=(True,), use_container_width=True):
        demo = st.text_input("Enter your question?","Request to access the Dashboard?",disabled=st.session_state.get("disabled", True))
        with st.expander("Answer"):
            handle_user_input2(demo)
    if st.button("Facing issues with SOP/guidelines", key="demo2",on_click=disable, args=(True,), use_container_width=True):
        demo2 = st.text_input("Enter your question?","Even after following the SOP/ guidelines I am still facing issue?",disabled=st.session_state.get("disabled", True))
        with st.expander("Answer"):
            handle_user_input2(demo2)
    # if st.button("if Requested duration is exhausted?", key="demo3",on_click=disable, args=(True,), use_container_width=True):
    #     demo3 = st.text_input("Enter your question?","What will happen if the Requested duration is exhausted?",disabled=st.session_state.get("disabled", True))
    #     with st.expander("Answer"):
    #         handle_user_input2(demo3)



    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    directory_path = "../contents" 
    if os.path.isdir(directory_path):
        documents = load_docs_from_folder(directory_path)
        if documents:
            persist_directory = "chroma_db"
            text_chunks = get_text_chunks(documents)
            vector_store = create_vectorstore(
                persist_directory, text_chunks, user_question)
            st.session_state.conversation = create_conversation_chain(
                vector_store)
        else:
            st.warning("No text files found in the folder.")
    else:
        st.error("Invalid folder path. Please enter a valid path.")


if __name__ == '__main__':
    main()