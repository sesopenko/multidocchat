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
        max_length=500,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline)

    template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
    Answer the question below from context below :
    {context}
    {question} [/INST] </s>
    """

    question_p = """Which companies announced their mergers"""
    context_p = """ In a landmark fusion of tech titans, Cybervine and QuantumNet announced their merger today, promising to redefine the digital frontier with their combined innovation powerhouse, now known as CyberQuantum."""
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"question": question_p, "context": context_p})
    print(response)


if __name__ == '__main__':
    main()
