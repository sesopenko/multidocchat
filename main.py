# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.schema.document import Document

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain


async def main():
    all_splits = get_game_splits()
    llm = get_llm()





    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectordb = Qdrant.from_documents(documents=all_splits, embedding=embeddings, location=":memory:",
         prefer_grpc=True,
         collection_name="my_documents",
     )  # Local mode with in-memory storage only

    retriever = vectordb.as_retriever()

    # Example template: <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
    prompt_template = """
            ### [INST] 
            Instruction: Answer the question based on your 
            Pathfinder 2nd Edition game and world knowledge. Here is context to help:

            {context}

            ### QUESTION:
            {question} 

            [/INST]
            """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        } | llm_chain
    )
    result = rag_chain.invoke("What is a short description of the Elite Viewing Room?")
    print(result)

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     verbose=True
    # )
    #
    # def run_my_rag(qa, query):
    #     print(f"Query: {query}\n")
    #     result = qa.run(query)
    #     print("\nResult: ", result)
    #
    # query = """ What's in the Elite Viewing Room? """
    # run_my_rag(qa, query)


def get_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model_4bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                      quantization_config=quantization_config, )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=1024,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )



    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm


def get_game_splits():
    data_location = "/home/sean/src/github.com/sesopenko/foundrydbscraper/generated"
    loader = DirectoryLoader(data_location,
                             glob="**/*.html",
                             show_progress=True,
                             # use_multithreading=True,
                             loader_cls=BSHTMLLoader,
                             )
    docs = loader.load()
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_splits = []
    for doc in docs:
        doc_splits = html_splitter.split_text(doc.page_content)
        html_splits += doc_splits
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=30)
    all_splits = recursive_splitter.split_documents(html_splits)
    return all_splits


if __name__ == '__main__':
    asyncio.run(main())
