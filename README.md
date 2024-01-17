# sesopenko/multidocchat

MLL query system for the Pathfinder game lore world. Uses a GPU memory database for the embedding so that 10k documents
can be queried efficiently using an MLL.

## Requirements

My own environment has the following:

* [a dataset of html files](https://github.com/sesopenko/pfw_scraper) 
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

Set your `DOC_LOC` environment variable to the location of your game world html files. You can scrape them with
[pfw_scraper](https://github.com/sesopenko/pfw_scraper).

## Licensed Apache V2

[LICENSE.txt](LICENSE.txt) should be included when distributing this software.

## Copyright

Copyright (c) Sean Esopenko 2024, all rights reserved.