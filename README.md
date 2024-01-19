# ![plot](./apple.png) APPLS
This repository contains the code for POMME score from APPLS: : [A Meta-evaluation Testbed for Plain Language Summarization](https://arxiv.org/pdf/2305.14341.pdf).
## Updates

## POMME score
1. Begin by cloning this repository. Then, install the necessary packages listed in requirements.txt by creating a Conda environment:
```
conda create --name pomme --file requirements.txt
```

2. Place your test.csv file into the ./data/ directory. This file needs to have two columns: 'id' and 'perturbed_text'. The 'id' column should contain the text's identifier, while 'perturbed_text' is the specific text you wish to analyze.

3. Navigate to the ./pomme/run_pomme.sh script. Here, you should modify the script to specify the file name and choose the desired in-domain and out-domain models for your evaluation. 

4. The default reference dataset is located in ./data/ref/ and originates from the [CELLS dataset](https://github.com/LinguisticAnomalies/pls_retrieval). Feel free to replace this with another dataset of your choice for reference purposes. An example CSV file is provided in ./data/ as a guide for running the code.

## Citation
```
@article{guo2023appls,
  title={APPLS: A Meta-evaluation Testbed for Plain Language Summarization},
  author={Guo, Yue and August, Tal and Leroy, Gondy and Cohen, Trevor and Wang, Lucy Lu},
  journal={arXiv preprint arXiv:2305.14341},
  year={2023}
}
```
