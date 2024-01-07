# LexRank-based Text Summarization System

Author: Tatjana Chernenko, 2018

This project focuses on enhancing the LexRank-based text summarization system by incorporating semantic similarity measures from the ECNU system.

Inspiration:

- G. Erkan, D. R. Radev. (2014). LexRank: Graph-based Lexical Centrality as Salience in Text Summarization
- Junfeng Tian et al.(2014). ECNU at SemEval-2017 Task 1: Leverage Kernel-based Traditional NLP features and Neural Networks to Build a Universal Model for Multilingual and Cross-lingual Semantic Textual Similarity

Results: [paper]()

## Overview

The LexRank-based text summarization system employs a stochastic graph-based method to compute the relative importance of textual units for extractive multi-document text summarization. This implementation initially utilizes cosine similarity between sentences as a key metric.
In this model, a connectivity matrix based on intra-sentence cosine similarity is used as the adjacency matrix of the graph representation of sentences.
## Objective

The objective is to explore the impact of replacing cosine similarity with a combination of features from the ECNU system, known for its semantic similarity measure. This modification aims to improve the summarization effectiveness of the LexRank approach.

## Features Implemented

- LexRank Implementation:
  - Sentence representation using TF·IDF metrics
  - Cosine similarity matrix generation
  - Sentence centrality computation
  
- Implementation of ECNU Features and Learning Algorithms:
  - Traditional NLP Module extraction
  - Deep Learning Module training
  - Ensemble Module for final score aggregation

- Modified LexRank (LexRank with ECNU):
  - Replacing cosine similarity with ECNU features and learning algorithm

## Evaluation Data and Results

The evaluation is conducted using the DUC2003 summarization evaluations dataset. Performance is measured using ROUGE (Recall-Oriented Understudy for Gisting Evaluation), providing insights into the summarization quality.

### Performance Metrics

| Metric           |
|------------------|
| ROUGE-1 Recall   |
| ROUGE-1 Precision|
| ROUGE-1 F-score  |
| ROUGE-su4 Recall | 
| ROUGE-su4 Precision |
| ROUGE-su4 F-score| 

## Conclusion and Future Work

Both LexRank and Modified LexRank showcase competitive ROUGE scores 
(61.7% ROUGE-su4 on the three multi-document clusters for the short 
summaries of three sentences), indicating good summarization capability. 
The modified version, leveraging ECNU features, allows us to achieve 62.87% 
average ROUGE-su4. The modified system demonstrates potential for improvement, 
particularly with further experiments and enhancements like increased sentence limits and the inclusion of a reranker module.

# Run

## 1. Getting Started

### 1.1. Prerequisites

- python3

- Get a copy of the project on your local machine for development and testing purposes:
```
git clone https://gitlab.cl.uni-heidelberg.de/chernenko/automatic_textsummarization_ws_17_18.git
cd automatic_text_summarization
```

Download https://stanfordnlp.github.io/CoreNLP/ and put the unzipped folder to the data folder.

#### 1.2. Automatic installation of dependencies:

```
sh download.sh 
```
#### 1.3. Manual installation of dependencies:

```
pip install -U scikit-learn
pip install pyprind
pip install python-jsonrpc
pip install gensim
pip install selenium
pip install nltk
pip install lxml
pip install breadability
pip install beautifulsoup4
```


* lanch the stanford CoreNLP:

```
cd ./data/stanford-corenlp-full-2015-12-09/stanford-corenlp-full-2015-12-09

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```

After this, you will find stanfordCoreNLP server at http://localhost:9000/

Write all new commands in a new terminal window.

## 2. Run the demo

* A. You have the possibility to try a standard Implementation of LexRank on demo data (one DUC20013 Task 2 multi-document cluster, copressed in one file).

After installing the program (Section 1.1 and 1.2 (auto) /1.3(manually)) run the following commands in your terminal 
(we assume that you are in the automatic_text_summarization folder):

```
cd demo
sh demo.sh
```

You find the output summary in ./demo folder.

* B. You also can try a standard version of LexRank for web-pages:

```
cd demo
python lex_rank_demo.py my-lex-rank --length=10 --url=http://en.wikipedia.org/wiki/Automatic_summarization
```

* C. If you want to test the modified version of LexRank, please note, that it will take from 6-7 hours just for one of three document clusters. 
Please follow the instructions in Section 3, part "Lounch an improved version of multi-document LexRank".

## 3. Install the whole version and run


- **Get & preprpocess test data** from the set of documents of Task 2 DUC2003:

```
cd ../ # home directory of the project
cd lib
mkdir test_data
cd ../bin
python get_test_data.py
```

Open stanfordCoreNLP server in browser (please check that you have run sh download sh in (Section 1.2 for automatic installation) 

or have lancheded the stanford CoreNLP manually (Section 1.3 for manually installation) before running this command:

```
http://localhost:9000/
```

- Lounch **a standard version of multi-document LexRank** without re-ranker (length of output files = 3 sentences).

(Find the output summaries in ./output/summaries_standard)

In a new terminal window run:

```
cd ..
mkdir output
cd output
mkdir summaries_standard
cd ../
sh shell_standard.sh
```

- Lounch **an improved version of multi-document LexRank** without re-ranker (length of output files = 3 sentences):

(Find the output summaries in ./output/summaries_improved)

```
cd output
mkdir summaries_improved
cd ../
sh shell_en.sh
```

- **Evaluate** the output:

```
cd ./bin
python pyrouge.py
```

- **NOTE:**
The standard version of the LexRank takes some minutes to summarize all the test data (texts from the Task 2 DUC2003).

The improved version of LexRank takes from 6 hours (depending on your system) to summarize just one of these texts.

The project provides all the summaries of the standard LexRank for the test data and summaries of three texts (30020, 30024 and 31010) of the improved LexRank, which are used for the evaluation.

### Try other features (for future):

The project considers Gradient Boosting Regression learning algorithm and two features of ECNU system: 
WeightednGramOverlapFeature(type='lemma') and BOWFeature(stopwords=False).

**If you want to add other features:**
This project provides basic architechture for adding ather modules, features and learning algorithms of ECNU system in future, wgich are now under construction. 

For future: 

Open ./sts\_model.py file, choose some features/combinations and add them to the file ./bin/lex_rank_en.py to the class LexRankSummarizer(AbstractSummarizer) (line 176 in lex_rank_en.py) LexRankSummarizer(AbstractSummarizer)

Note: This could increase the time of running the programm.

### Possible issues:

If you have any issues with opening the files, check the style of strings (depends on your system) for the path to the files in *.py files in bin folder.

If you have any issues with Datasets in data directory, you can load them yourself:

* download the STSBenchmark Dataset:

```
mkdir data
cd data
```
Dowload data from 
```
http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
```

Unpack the file Stsbenchmark.tar.gz

Remove Stsbenchmark.tar.gz

* download the stanford CoreNLP 3.6.0

```
http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
```

Then unzip stanford-corenlp-full-2015-12-09.zip

## 3. Authors

* **Tatjana Chernenko** 

## 4. License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](https://gitlab.cl.uni-heidelberg.de/chernenko/automatic_textsummarization_ws_17_18/blob/master/LICENSE) file for details

## 5. Acknowledgments

Inspiration:

[1][G. Erkan, D. R. Radev. (2014). LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://jair.org/index.php/jair/article/view/10396)

[2][Daniel Cer et al. (2017). SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation](http://nlp.arizona.edu/SemEval-2017/pdf/SemEval001.pdf)

[3][Junfeng Tian et al.(2014). ECNU at SemEval-2017 Task 1: Leverage Kernel-based Traditional NLP
features and Neural Networks to Build a Universal Model for Multilingual
and Cross-lingual Semantic Textual Similarity](http://www.aclweb.org/anthology/S17-2028)

[Single-document LexRank](https://github.com/miso-belica)

[Semantic Textual Similarity (STS)](https://github.com/rgtjf/Semantic-Texual-Similarity-Toolkits)


For further details, including implementation specifics, evaluation results, and analysis, refer to the project documentation and associated research papers.
