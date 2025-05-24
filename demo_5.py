from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from googlesearch import search
from langchain_community.document_loaders import WebBaseLoader

from langchain_nomic.embeddings import NomicEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from flask import Flask, request, render_template


def relevance(vector, query):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,max_tokens=10000)
    #llm = ChatOllama(model="llama3", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n Provide the binary score with no premable or explanation."),
        ("user", "Context: Here is the retrieved document: \n\n {context} \n\n Here is the user question: {input}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    print(f'relevant?: {response["answer"]}')
    return response["answer"]
        

def summarize(vector, query):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,max_tokens=10000)
    #llm = ChatOllama(model="llama3", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context and summarize how to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise"),
        ("user", "Question: {input} Context: {context} Answer:")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

def grader(summary, query):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,max_tokens=10000)
    #llm = ChatOllama(model="llama3", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Provide no preamble or explanation."),
        ("user", "Here is the answer:\n ------- \n{summary} \n ------- \nHere is the question: {query}")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(input={"query": query, "summary": summary})
    print(f'good?: {response}')
    return response

global_vectors = []
sources = []

def process():
    if request.method == 'POST':
        query = request.form['query']
        num_results = request.form['num']
        follow_up_query = request.form.get('new_user_input', None) 
        try:
            num_results = int(num_results)
        except ValueError:
            return render_template('index.html', result=["No results found"])
        urls = search(query, num_results=num_results, unique=True)
        loader = WebBaseLoader(list(urls))
        docs = loader.load()
        #embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        #text_splitter = RecursiveCharacterTextSplitter()
        #documents = text_splitter.split_documents(docs)
        
        #global_vectors.append(FAISS.from_documents(documents, embeddings))
        results = []
        for doc in docs:
            vector = FAISS.from_documents([doc], embeddings)
            if relevance(vector, query) == "yes":
                global_vectors.append(vector)
                sources.append(doc.metadata['source'])
                summary = summarize(vector, query)
                if grader(summary,query) == "yes":
                    results.append(f"url: {doc.metadata['source']}:\n {summary}")
        return render_template('index.html', result=results)
    return render_template('index.html', result=["No results found"])

def process_follow_up_input():
    if request.method == 'POST':
        results = []
        query = request.form['query_2']
        for vector, url in zip(global_vectors,sources):
            summary = summarize(vector, query)
            if grader(summary,query) == "yes":
                    results.append(f"url: {url}:\n {summary}")
        return render_template('index.html', result=results)
    return render_template('index.html', result=["No result found"])

def handle_reset():
    global global_vectors
    global sources
    global_vectors = []
    sources = []
    return render_template('index.html', result=["No result found"])