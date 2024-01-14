import os
import torch
import transformers
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  pipeline
)
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.vectorstores import Qdrant

def pathquery():

    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    model_config = transformers.AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pading_size = "right"

    #################################################################
    # bitsandbytes parameters
    #################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    #################################################################
    # Set up quantization config
    #################################################################
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerating training with bf16")
            print("=" * 80)
            bnb_4bit_compute_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    #################################################################
    # Load pre-trained config
    #################################################################
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

    print(print_number_of_trainable_model_parameters(model))

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1000,
    )

    doc_splits = get_game_splits()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    # db = Qdrant.from_documents(documents=doc_splits, embedding=embeddings, location=":memory:",
    #      prefer_grpc=True,
    #      collection_name="my_documents",
    #  )  # Local mode with in-memory storage only
    db = FAISS.from_documents(doc_splits, embeddings)

    retriever = db.as_retriever()

    prompt_template = """
        ### [INST] 
        Instruction: Answer the question based on your 
        Pathfinder 2nd Edition game and world knowledge. Here is context to help:

        {context}

        ### QUESTION:
        {question} 

        [/INST]
        """
    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        } | llm_chain
    )
    result = rag_chain.invoke("Who is the boss of the Library?")
    print(result["text"])

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
    pathquery()
