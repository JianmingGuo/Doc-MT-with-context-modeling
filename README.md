# Doc-MT-with-context-modeling
This repo aims to perform document-level NMT by context modeling with hierarchical shallow attention structure

## Requirements
This implementation is based on the repo [NMT_GAN](https://github.com/ZhenYangIACAS/NMT_GAN) 

    Tensorflow >= 1.13
    Python versin >= 3.6
    CUDA & cudatoolkit >= 10.1

> conda install --yes --file requirements.txt

## Process the corpus
To ensure the quality of the input corpus. Tokenization, BPE, clean and shuffle may be necessary. The relate script is in the file ```corpus/process.sh```

> bash corpus/process.sh

Given the purpose of the repo is Doc-MT, the corpus should be document-level too. The format of the parallel corpus can be:

    #train.src
    sentence1 || sentence2 || ... || sentencen  #doc1
    sentence1 || sentence2 || ... || sentencem  #doc2

    #train.dst
    sentence1 || sentence2 || ... || sentencen  #doc1
    sentence1 || sentence2 || ... || sentencem  #doc2

To ensure the comparability of our experimental results, we use the same corpus as the [selective-attn](https://github.com/sameenmaruf/selective-attn/tree/master/data)

## Train the Model

The yaml file which defines the parameters can be built according to your needs. Some examples are in ```configs/```

> python train_con_ijcnn.py -c YAML_FILE

## Evaluate the Model

> python evaluate.py -c YAML_FILE  
> bash evaluate.sh
