import datetime
import glob
import os
import openai
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
# Set up
import random
import time
from PyPDF2 import PdfReader

# os.environ["http_proxy"] = "http://127.0.0.1:33210"
# os.environ["https_proxy"] = "http://127.0.0.1:33210"
#os.environ["OPENAI_API_KEY"] = "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"
st.title("PDF文档对话聊天机器人")
current_date = datetime.datetime.now().date()

if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
embeddings = OpenAIEmbeddings()
template = """
请使用以下上下文回答最后的问题。如果你不知道答案，那就说你不知道，不要试图编造答案。最多使用三句话。请尽可能地简洁。回答结束时请说“请问还有什么可以帮您”。请使用中文来回答问题。
{context}
问题: {question}
有帮助的答案:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
#title = st.sidebar.text_input('此处填入API-KEY', 'API-KEY')
def set_api_key():
    user_api_key = st.sidebar.text_input(
        label="#### 在此填入API key填写完成后回车👇",
        placeholder="Paste your openAI API key, sk-",
        type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        openai.api_key = user_api_key
        # try:
        #     response = openai.Completion.create(
        #     engine="text-davinci-002",
        #     prompt="Translate the following English text to French: '{}'",
        #     max_tokens=60,
        #
        #  )
        st.sidebar.write(" 填入成功下一步选取pdf")

    # except Exception as e:
    #     st.sidebar.write("无效的API key，请重新填入", e)
#openai.api_key = "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"


def load_db(pdf_list, chain_type, k):
    # load documents
    #loader = PyPDFLoader(file)
    #documents = loader.load()
    # split documents
    #embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    text = ""
    for pdf in pdf_list:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    chunks = text_splitter.split_text(text)
    #docs = text_splitter.split_text(text)
    # define embedding
    # create vector database from data
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    #db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # define retriever
    retriever = knowledge_base.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

# Load all PDF files in the specified directory
# pdf_files = glob.glob('F:\项目学习\chatpdf\*.pdf')
# qas = {os.path.basename(file): load_db(file, "stuff", 4) for file in pdf_files}

chat_history = []

# Build prompt




# selected_file = st.sidebar.selectbox("选择文件", list(qas.keys()))
# qa = qas[selected_file]
#
# st.sidebar.title("聊天历史")
# for exchange in chat_history:
#     st.sidebar.markdown(f"Q: {exchange[0]}\nA: {exchange[1]}\n---")

def main():
    st.sidebar.title("请选择PDF文件")
    pdf_list = st.sidebar.file_uploader("一次性选择一个或者多个PDF文件", type="pdf", accept_multiple_files=True)
    if pdf_list != []:
        st.sidebar.write("文件载入成功，现在可以进行文档问答")
    print(pdf_list)
    # st.sidebar.title("聊天历史")
    # for exchange in chat_history:
    #     st.sidebar.markdown(f"Q: {exchange[0]}\nA: {exchange[1]}\n---")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        if pdf_list != []:
            # st.markdown(f"答案: {result['answer']}")
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                qa = load_db(pdf_list, "stuff", 4)
                result = qa({"question": prompt, "chat_history": chat_history})
                chat_history.append((prompt, result["answer"]))
                assistant_response = result['answer']
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})


# if st.button("提交"):
#     if user_input:
#         result = qa({"question": user_input, "chat_history": chat_history})
#         chat_history.append((user_input, result["answer"]))
#         st.markdown(f"答案: {result['answer']}")
if __name__ == '__main__':
    set_api_key()
    main()