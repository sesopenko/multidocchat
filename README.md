# sesopenko/multidocchat

## Original Source

This is originally from [Building a Multidocument Chatbot using Mistral 7B, Qdrant, and Langchain](https://blog.stackademic.com/building-a-multidocument-chatbot-using-mistral-7b-qdrant-and-langchain-1d9982186736)
by Vardhanam Daga.

## Requirements

My own environment has the following:

* conda (or miniconda)
* Nvidia 30 series or later gpu
* Debian 12 Linux
* Nvidia official drivers (.sh installation)
* cuda toolkit installed 12.1 and 11.4
* cuda 12 or later drivers installed

## Setup

```bash
conda create multidoc
conda activate multidoc
pip install -r requirements.txt
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

## Further Reading

* https://python.langchain.com/docs/use_cases/code_understanding

