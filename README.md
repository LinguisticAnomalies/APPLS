# ![plot](./apple.png) APPLS
Welcome to the repository for **APPLS**, which provides code for constructing a testbed and performing evaluation for the paper:  
[**APPLS: Evaluating Evaluation Metrics for Plain Language Summarization**](https://aclanthology.org/2024.emnlp-main.519/).

---

## Updates
- **Nov 24, 2024**: Added lexical features and LLM-based evaluation codes.
- **Sep 20, 2024**: Our paper has been accepted to EMNLP 2024 (Main Conference)!

---

## Overview

The goal of this repository is to assess how well existing metrics capture the multiple criteria of PLS. We identify four criteria, informed by prior work, that a PLS metric should be sensitive to: **informativeness**, **simplification**, **coherence**, and **faithfulness**. We introduce a set of perturbations to probe metric sensitivity to these criteria, where each perturbation is designed to affect a single criterion with ideally minimal impact to others.

![plot](./perturbation_example.png) Example perturbations for criteria in APPLS. Original text comes from the CELLS study.


## Perturbing Your Dataset

If you’d like to perturb your dataset to see how well existing metrics capture your text, please follow these steps to build the testbed using your dataset:

1. **Prepare Your Dataset**:
   - The input dataset should be in a CSV file with two columns named “id” and “reference text”, representing the index of the text and the original text to be perturbed.

2. **Run the Code**:
   - For tasks other than entity_swap, use the following code:
     ```bash
     python main.py \
            --task simplification \
            --input_file $input_file \
            --input_path $DATA_PATH \
            --output_path $DATA_PATH
     ```

   - For the entity_swap task, follow these additional steps:
     1. Follow the [claim generation repository instructions](https://github.com/allenai/scientific-claim-generation) to install the necessary packages.
     2. Change the `scientific_claim_folder` in `factual_consistency_entity_swap.py`.
     3. Run `main_entity_swap.py`.

## Evaluation

This project analyzes **8 established evaluation metrics**, including the **5 most commonly reported metrics** in ACL'22 summarization and generation papers. Additionally, we assess **5 lexical features** associated with text simplification and perform **LLM-based evaluations**.


### Lexical Features
To evaluate lexical features, use the script: `./evaluation/lexical_features/eval_difficulty.py`

#### Specificity Calculation
For the **specificity** metric among lexical features, we utilize **Speciteller**, a domain-agnostic tool to compute the specificity of terms in a paragraph. You can find Speciteller here: [https://github.com/jjessyli/speciteller](https://github.com/jjessyli/speciteller).


### LLM-Based Evaluation
To conduct LLM-based evaluations, execute the following script: `./evaluation/gpt/run_gpt.sh`

#### Instructions
1. Modify the code path in `run_gpt.sh` to match your environment.
2. Enter your API key in the `utils` file to enable GPT-based evaluations.


## POMME score
1. Begin by cloning this repository. Then, install the necessary packages listed in requirements.txt by creating a Conda environment:
```
conda create --name pomme --file requirements.txt
```

2. Place your test.csv file into the ./data/ directory. This file needs to have two columns: 'id' and 'perturbed_text'. The 'id' column should contain the text's identifier, while 'perturbed_text' is the specific text you wish to analyze.

3. Navigate to the ./pomme/run_pomme.sh script. Here, you should modify the script to specify the file name and choose the desired in-domain and out-domain models for your evaluation. 

4. The default reference dataset is located in ./data/ref/ and originates from the [CELLS dataset](https://github.com/LinguisticAnomalies/pls_retrieval). Feel free to replace this with another dataset of your choice for reference purposes. An example CSV file is provided in ./data/ as a guide for running the code.

## References

Guo, Y., et al. (2022). CELLS: A Comprehensive Benchmark for Evaluation of Language Model Outputs.

For further assistance or questions, feel free to open an issue in this repository.

## Citation
```
@inproceedings{guo-etal-2024-appls,
    title = "{APPLS}: Evaluating Evaluation Metrics for Plain Language Summarization",
    author = "Guo, Yue  and
      August, Tal  and
      Leroy, Gondy  and
      Cohen, Trevor  and
      Wang, Lucy Lu",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.519",
    pages = "9194--9211",
    abstract = "While there has been significant development of models for Plain Language Summarization (PLS), evaluation remains a challenge. PLS lacks a dedicated assessment metric, and the suitability of text generation evaluation metrics is unclear due to the unique transformations involved (e.g., adding background explanations, removing jargon). To address these questions, our study introduces a granular meta-evaluation testbed, APPLS, designed to evaluate metrics for PLS. We identify four PLS criteria from previous work{---}informativeness, simplification, coherence, and faithfulness{---}and define a set of perturbations corresponding to these criteria that sensitive metrics should be able to detect. We apply these perturbations to extractive hypotheses for two PLS datasets to form our testbed. Using APPLS, we assess performance of 14 metrics, including automated scores, lexical features, and LLM prompt-based evaluations. Our analysis reveals that while some current metrics show sensitivity to specific criteria, no single method captures all four criteria simultaneously. We therefore recommend a suite of automated metrics be used to capture PLS quality along all relevant criteria. This work contributes the first meta-evaluation testbed for PLS and a comprehensive evaluation of existing metrics.",
}

```
