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

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.schema.document import Document

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import HTMLHeaderTextSplitter


def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model_4bit = AutoModelForCausalLM.from_pretrained( model_id, device_map="auto",quantization_config=quantization_config, )
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

    mna_news = """Before their ancient clash with humanity devastated their civilization, serpentfolk were masters of a sprawling underground empire. Few serpentfolk survive today; their power is shattered, their god Ydersius decapitated (although not quite slain). The cunning, intelligence, and magical abilities of serpentfolk have diminished from their ancient heights, and most are born without these boons. Those serpentfolk who retain their ancestry’s legacy of intelligence and magic are known as zyss, and they look down upon their more numerous kindred with a mix of disdain and shame. They see these offshoots as a curse on their kind, resulting from their god’s decapitation and the pandemonium during the fall of their underground empire, and have dubbed them aapoph, meaning “chaos made flesh.”

Today, the central realm of the Darklands retains the old name of the serpentfolk empire that once dominated this region—Sekamina. This name is also the source of the serpentfolk’s Aklo title, sekmin, which they are often called in ancient texts. Sekamina itself retains very little of the serpentfolk’s legacy, its mantle of rule having passed on to others like drow, ghouls, gugs, and deep gnomes. Yet in remote reaches of this dangerous realm, the ruins of serpentfolk cities still stand. Within, a great many serpentfolk sleep in torpor in secluded vaults, with only a few cells awake to enact their schemes. In addition, a small number of serpentfolk settlements dot Golarion’s surface, most of them in humid, remote jungles, far-flung islands, or caverns close to the surface. It’s rare for such a settlement to number more than a few dozen serpentfolk. They rely primarily on slaves to build their power bases, to defend them, and to perform essentially all the practical functions of their society. This includes providing food, crafting goods, and tending to the serpentfolk’s every need.

Zyss serpentfolk are megalomaniacal geniuses with dreams of returning to their place of dominance, though modern serpentfolk have few means of accomplishing this goal. Many of their plans hinge on resurrecting Ydersius, their decapitated god. His headless body still thrashes about, mindless, in the Darklands, waiting to be reunited with his lost skull. Serpentfolk numbers are so small that reclaiming their dominance seems a distant dream, especially since their reproduction is slow. Though a parent can birth a dozen young at once, the gestation period lasts up to a decade, and the likelihood that even one will be zyss is low. There’s no telling whether a child will be zyss or aapoph, regardless of parentage. A coveted zyss child is just as likely to arise from aapoph parents as from two zyss, and every serpentfolk colony has someone in charge of sorting the young, identifying the earliest signs of intelligence in them.

Though the number of zyss is small in serpentfolk colonies, bringing in more zyss isn’t necessarily desirable. A serpentfolk conclave with just a few zyss is functional, but one with a large number becomes fractious. Cults and societies form, all pursuing their own passions and politics, with scheming and backstabbing running rampant. A powerful priest may be able to bring other zyss to heel, but many zyss question why a priest should be in charge if their god is dead. Zyss thrive on selfish desires for hedonistic pleasure and adulation. They feel no love for others, even their offspring. Thriving on decadence, they crave receiving expensive gifts, gorging themselves on massive meals, and pursing arts such as music, poetry, or sculpture. Even more academic hobbies, like the study of magic or warfare, take an artistic bent, like carefully designing colorful illusions or memorizing epic poems about renowned wars. Each zyss believes themself to have more refined tastes than their peers.

    """

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
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=30)
    all_splits = recursive_splitter.split_documents(html_splits)
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    # all_splits = text_splitter.split_documents(docs)

    # documents = [Document(page_content=mna_news, metadata={"source":"local"})]
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    # all_splits = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectordb = Qdrant.from_documents(documents=all_splits, embedding=embeddings, location=":memory:",
         prefer_grpc=True,
         collection_name="my_documents",
     )  # Local mode with in-memory storage only

    retriever = vectordb.as_retriever()

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

    query = """ What's in the Elite Viewing Room? """
    run_my_rag(qa, query)

if __name__ == '__main__':
    main()
