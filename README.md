# A dual-encoding system for dialect classification @ VarDial 2020

This repository contains the source-code of the [dual-encoding ensemble model](https://www.aclweb.org/anthology/2020.vardial-1.20/) built by UAIC team for VarDial2020 evaluation campaign.

## Romanian Dialect Identification
Romanian Dialect Identification (RDI) is a task introduced at the VarDial evaluation campaign<sup>[1](#vardial2020report)</sup> in 2020.

This is a closed task  aimed at classifying a text sentence as whether it pertains to the Moldavian idiom or one of the other idioms of Romanian language (although an idiom is less than a dialect, in the following we will refer to the idioms as being dialects).

For this task, our team proposes an ensemble model which focuses on exploiting the known differences in writing between the two dialects, and on emphasizing such differences for classification.

## Model

Our model is built upon the architectures of the Word-Based Convolutional Neural Network<sup>[2](#elaraby2018deep)</sup>, and the Meta-classifier<sup>[3](#malmasi2017arabic)</sup> --- both developed for Arabic Dialect Identification.

We developed two versions of the model --- one for each submission of the results. The second version was built from the first by adding supplementary layers depicted by the innermost rectangle in Figure~\ref{fig:architecture}.

### Core Idea

At the core of our model lies the intuition that the peculiarities of each of the two Romanian dialects can be detected by spotting the differences in writing.

**E.g.**: The term for _this much_ is written in Moldavian dialect as _atît_ and in Romanian dialect as _atât_. Also Moldavian dialect uses the outdated term _odaie_ for _room_, whereas in Romanian dialect the term _cameră_ is preferred.

### Training
- Each input sentence is transformed into two TF-IDF encodings: one for Romanian and one for Moldavian,
- The encodings are merged into a single tensor by column-wise concatenation,
- The resulting tensor is sent to the deep-learning model which learns to classify the sample according to its label.

The model was trained for 25 epochs with a batch size of 32 samples for each submission.

### Hyper-Parameters

| 1D convolution         | Dropout rate | Output dimensions  | Adam optimizer       |
|------------------------|--------------|--------------------|----------------------|
| Number of filters: 512 |          0.3 | Dense layer 1: 128 | Learning rate: 0.001 |
| Kernel size: 7         |              | Dense layer 2: 32  | Decay rate 1: 0.9    |
|                        |              |                    | Decay rate 2: 0.999  |

## Results

The results presented herein are obtained using the highest performing weights for each run.

### Accuracy scores

| Submission | Training score | Validation score | Evaluation score |
|------------|----------------|------------------|------------------|
| 1          | 0.8091         | 0.7988           | 0.5553           |
| 2          | 0.8220         | 0.8106           | 0.5402           |

## Conclusions

The design decision to discriminate based on textual marks leads to poor performance of the model. Furthermore, the data from the evaluation set does not confirm the hypothesis; i.e. the evaluation set contains samples that are labeled as pertaining to Moldavian dialect but use the writing marks of the Romanian dialect.

## Citation

If you want to cite our paper you can use the following `BibTex` entry:
```
@inproceedings{rebeja2020dual,
  title={A dual-encoding system for dialect classification},
  author={Rebeja, Petru and Cristea, Dan},
  booktitle={Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects},
  pages={212--219},
  year={2020}
}
```

or the `ISO 690` style:
```
REBEJA, Petru; CRISTEA, Dan. A dual-encoding system for dialect classification. In: Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects. 2020. p. 212-219.
```
## References & Footnotes
<a name="vardial2020report">1</a>: Andrei Butnaru and Radu Tudor Ionescu. “MOROCO: The Moldavian and Romanian Dialectal Corpus”. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019, pp. 688–698.

<a name="elaraby2018deep">2</a>: Mohamed Elaraby and Muhammad Abdul-Mageed. “Deep models for arabic dialect identification on benchmarked data”. In: Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2018). 2018, pp. 263–274.

<a name="malmasi2017arabic">3</a>: Shervin Malmasi and Marcos Zampieri. “Arabic dialect identification using iVectors and ASR transcripts”. In: Proceedings of the Fourth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial). 2017, pp. 178–183.
