import sys
import os

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import  HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS

config = {
    **dotenv_values('.env')
}

QUERY = None
chat_history = []


loader = DirectoryLoader(
    config.get('DATA_DIR'),
    glob="**/*",
    loader_cls=TextLoader,
    use_multithreading=True,
    loader_kwargs={'autodetect_encoding': True}
)

files = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
all_splits = splitter.split_documents(files)


def embed_index(split_list, embed_fn, index_store):
    faiss_db = FAISS.from_documents(split_list, embed_fn)
    if os.path.exists(index_store):
        local_db = FAISS.load_local(index_store, embed_fn, allow_dangerous_deserialization=True)
        local_db.merge_from(faiss_db)
        local_db.save_local(index_store)
    else:
        faiss_db.save_local(folder_path=index_store)


embedding_func = OpenAIEmbeddings(
    openai_api_key=config.get('OPENAI_API_KEY'),
    model="text-embedding-3-large"
)

embed_index(all_splits, embedding_func, config.get('DATA_STORE'))

vectorstore = FAISS.load_local(
    config.get('DATA_STORE'),
    embedding_func,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

MODEL = ChatOpenAI(
    openai_api_key=config.get('OPENAI_API_KEY'),
    model="gpt-3.5-turbo",
    temperature=0
)


REPHRASE_TEMPLATE_STR = """Given the above conversation, generate a search query to \
look up in order to get information relevant to the conversation"""
rephrase_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", REPHRASE_TEMPLATE_STR)
])

retriever_chain = create_history_aware_retriever(MODEL, retriever, rephrase_prompt)

PROMPT_TEMPLATE_STR = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE_STR),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(MODEL, prompt)

conversational_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

if len(sys.argv) > 1:
    QUERY = sys.argv[1]

while True:
    if not QUERY:
        QUERY = input("\033[31m\r\nPrompt: \033[0m")
    if QUERY in ['quit', 'q', 'exit']:
        sys.exit()

    response = conversational_retrieval_chain.invoke({
        'chat_history': chat_history,
        "input": QUERY
    })

    chat_history.append(HumanMessage(QUERY))
    chat_history.append(AIMessage(response['answer']))
    print("\033[32m"+response['answer']+"\033[0m")

    QUERY = None
