# This is a sample Python script.
from datetime import datetime

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

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from qdrant_client import QdrantClient
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.schema.document import Document
from langchain_community.document_loaders.generic import GenericLoader
from langchain.text_splitter import Language
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

def main():
    llm = build_llm()

    repo_path = "/home/sean/src/github.com/kivy/kivy"

    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        exclude=[],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )

    documents = loader.load()
    print(f"Loaded {len(documents)}")

    # need to split into chunks for embedding and vector storage
    go_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    all_splits = go_splitter.split_documents(documents)
    print(f"split {len(all_splits)} documents")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    print("creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    print("creating vector db...")
    start_time = datetime.now()
    print(f"start time: {start_time.isoformat()}")
    vectordb = Qdrant.from_documents(documents=all_splits, embedding=embeddings, location=":memory:",
         prefer_grpc=True,
         collection_name="my_documents",
     )  # Local mode with in-memory storage only
    end_time = datetime.now()
    time_difference = end_time - start_time

    # Extract minutes and seconds from the time difference
    minutes, seconds = divmod(time_difference.total_seconds(), 60)
    print("Time span: {} minutes and {:.2f} seconds".format(int(minutes), seconds))

    print("getting retreiver...")
    retriever = vectordb.as_retriever()


    print ("creating qa...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )

    def run_my_rag(qa, query):
        print(f"Query: {query}\n")
        result = qa.run(query)
        print("\nResult: ", result)

    while True:
        query = input("Question:")
        run_my_rag(qa, query)





    # mna_news = """Vectora, a forward-thinking player in the tech startup ecosystem, has ushered in a new chapter by naming Priyanka Desai as its Chief Executive Officer. Desai, a renowned figure in the tech community for her groundbreaking work at Nexus Energy Solutions, takes the reins at Vectora to propel the company into a leader in sustainable technology. With an expansive vision and a stellar record, Desai emerged as the chosen leader after an extensive international search, reflecting the board's confidence in her innovative approach and strategic acumen.
    # This strategic appointment coincides with Vectora's most recent milestone--securing a transformative $200 million in funding aimed at accelerating its growth. Desai's illustrious career, highlighted by her success in scaling Nexus Energy Solutions to an industry vanguard, speaks to her exceptional leadership. "Priyanka is the embodiment of leadership with purpose, and her alignment with our core values is what makes this appointment a perfect match," expressed Anil Mehta, Vectora's co-founder. Desai's plans for Vectora not only encompass financial growth but also reinforce the company's pledge to environmental innovation.
    # Addressing the company after her appointment, Desai unveiled an ambitious roadmap to expand Vectora's R&D efforts and introduce groundbreaking products to reduce industrial carbon emissions. "I am energized to lead a company that is as committed to sustainability as it is to technological innovation," Desai shared, underscoring her commitment to combating the urgent challenges posed by climate change.
    # Desai's leadership style, characterized by her emphasis on inclusive growth and collaborative innovation, has been met with resounding approval from within Vectora's ranks and across the tech sector. Her drive for fostering a workplace where diverse ideas flourish has drawn particular admiration. "Priyanka brings a dynamic perspective to Vectora that will undoubtedly spark creativity and drive," commented Anjali Vaidya, a prominent technology sector analyst. "Her track record of empowering her teams speaks volumes about her potential impact on Vectora's trajectory."
    # As Desai takes charge, industry observers are keenly awaiting the rollout of Vectora's most ambitious endeavor yet--an AI-driven toolset designed to optimize energy management for a global clientele. With Desai at the wheel, Vectora stands on the precipice of not just market success, but also contributing a significant handprint to the global sustainability effort. The tech world is abuzz as Desai is set to officially step into her new role next week, marking a potentially transformative era for Vectora and the industry at large.
    #
    # """
    #
    # documents = [Document(page_content=mna_news, metadata={"source":"local"})]
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    # all_splits = text_splitter.split_documents(documents)
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {"device": "cuda"}
    # embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    #
    # vectordb = Qdrant.from_documents(documents=all_splits, embedding=embeddings, location=":memory:",
    #      prefer_grpc=True,
    #      collection_name="my_documents",
    #  )  # Local mode with in-memory storage only
    #
    # retriever = vectordb.as_retriever()
    #
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
    # query = """ What company is buyer and seller here """
    # run_my_rag(qa, query)


def build_llm():
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
        max_length=4092,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm


if __name__ == '__main__':
    main()
