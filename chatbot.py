import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import Chroma
import os,re
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder

def load_pdf(files):
    loader = PyPDFLoader(files)
    pages = loader.load_and_split()
    return pages


def list1(pages):
    l1 = []
    for i in pages:
        data = i.page_content
        l1.append(data)
    #print(l1)
    return l1

def fetch_questions(l1):
    questions = []
    pattern = r'Q\d+\..*?\?'

    # Extract questions
    for text in l1:
        matches = re.findall(pattern, text, re.DOTALL)
        questions.extend(matches)

    # # Print questions
    # for question in questions:
    #     print(question.strip())
    return questions

def get_pdf_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=20,
        length_function=len,
        separators=['\nQ','\n\n','\n']
    )
    chunks = text_splitter.create_documents(documents)
    #print(chunks)
    return chunks

def fetch_answers(l1):
    pattern = r'Answer:.*?(?=Q\d+\.|$)'
    answers = []

    # Extract answers
    for text in l1:
        matches = re.findall(pattern, text, re.DOTALL)
        answers.extend(matches)

    return answers

def m_chunks(chunks, answers):
    for chunk, answer in zip(chunks, answers):
        chunk_metadata = {"answer": answer}
        chunk.metadata = chunk_metadata
    return chunks

def persist(chunks, user_question):
    embeddings = OpenAIEmbeddings()
    persist_directory = "temp"
    retriever2 = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    retriever2.persist()
    quest = retriever2.similarity_search_with_score(user_question,k=1)
    return quest


# Text Files
def load_docs(text_file):
    documents = []
    # for text_file in text_files:
    content = text_file.getvalue().decode("utf-8")
    documents.append(content)
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.create_documents(documents)
    return chunks


def get_vectorstore(persist_directory, text_chunks, user_question):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    final_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    final_db.similarity_search_with_score(user_question,k=3)
    return final_db


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    print(st.session_state.chat_history)

    for i, message in list(enumerate(st.session_state.chat_history)):
        if i % 2 == 1:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    # if callable(st.session_state.conversation):
    #     if (user_question.lower().strip() in ['hi','hello','good morning','good afternoon','good evening']):
    #         question = user_question
    #     else:
    #         question = user_question + " Answer in step by step points only"
    #     response = st.session_state.conversation({'question': question})
    #     if st.session_state.chat_history is None:
    #         st.session_state.chat_history = []
    #     st.session_state.chat_history = response['chat_history'] + st.session_state.chat_history
    #     print(st.session_state.chat_history)
    #     for i, message in list(enumerate(st.session_state.chat_history)):
    #         if i % 2 != 0:
    #             val = ""
    #             for j in range(len(message.content)):
    #                 if message.content[j].isdigit() and j < len(message.content)-1 and message.content[j+1] == ".":
    #                     val += "\n" + message.content[j]
    #                 else:
    #                     val += message.content[j]
    #             message.content = val
    #             st.write(user_template.replace(
    #                 "{{MSG}}", message.content), unsafe_allow_html=True)
    #         else:
    #             st.write(bot_template.replace(
    #                 "{{MSG}}", message.content.split("Answer in step by step points only")[0]), unsafe_allow_html=True)
    # else:
    #     st.error("Conversation chain is not initialized. Please process documents first.")
        
def rerank_top_n(query,output,n_chunks):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    input_lst=[]
    for chunk in output:
        tup=(query,chunk[0].page_content)
        input_lst.append(tup)
    if len(input_lst) == 0:
        return []
    scores = model.predict(input_lst)
    total_data=zip(output,scores)
    reranked = sorted(total_data, key=lambda x: x[1],reverse=True)
    try:
        return reranked[:n_chunks]
    except:
        print(f"Value of n_chunks is greater than the number of input chunks. Reduce the number of n_chunks.")
        return reranked

def main():
    embeddings = OpenAIEmbeddings()
    os.environ["OPENAI_API_KEY"]= "YOUR OPENAI_API_KEY"
    st.set_page_config(page_title="Chat with multiple Files",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        # handle_userinput(user_question)
        db3 = Chroma(persist_directory="temp", embedding_function=embeddings)
        quest = db3.similarity_search_with_score(user_question,k=3)
        top = rerank_top_n(user_question,quest,1)
        if top[0][1]*10 > 60:
            print("Inside if")
            #print(quest[0][0].metadata['answer'])
            st.write(top[0][0][0].metadata["answer"])
        else:
            db4 = Chroma(persist_directory="chroma_db3", embedding_function=embeddings)
            # quest_llm = db4.similarity_search_with_score(user_question,k=3)
            # top_llm = rerank_top_n(user_question,quest_llm,1)
            # create conversation chain
            print("Inside else")
            st.session_state.conversation = get_conversation_chain(
                db4)
            handle_userinput(user_question)
        
    # elif():
    #     handle_userinput(user_question)
    # else: 
    #     pass

    with st.sidebar:
        st.subheader("Your documents")
        text_files = st.file_uploader(
            "Upload your text files here and click on 'Process'", accept_multiple_files=True)
        print(text_files)
        if st.button("Process"):
            with st.spinner("Processing"):
                print(text_files)
                for text_file in text_files:
                    print("=====================================",text_file)
                    if ".txt" in text_file.name:
                        # print("="*40, text_file[0].name)
                        documents = load_docs(text_file)

                        # Persist directory for Chroma database
                        persist_directory = "chroma_db3"

                        # get the text chunks
                        text_chunks = get_text_chunks(documents)

                        # Create Chroma database
                        # create vector store
                        vectorstore = get_vectorstore(persist_directory, text_chunks, user_question)


                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore)
                    elif (".pdf" in text_file.name):
                        print("==============",text_file)
                        with open(os.path.join("temp",text_file.name),"wb") as f:
                            f.write(text_file.read())
                        pdf_file_path = os.path.join("temp", text_file.name)
                        print(pdf_file_path)
                        pages = load_pdf(pdf_file_path)
                        l1 = list1(pages)
                        questions = fetch_questions(l1)
                        chunks = get_text_chunks(questions)
                        answers = fetch_answers(l1)
                        chunk_wm = m_chunks(chunks, answers)
                        quest = persist(chunk_wm, user_question)
                    
                    


if __name__ == '__main__':
    main()
