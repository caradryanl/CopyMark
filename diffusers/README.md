# Reproduction Guidelines

We provide complete scripts to reproduce the experiment results in our paper, which consist of three parts:

- Benchmark results (Section 4.2)

- Score distributions (Section 4.3)

- Case studies (Section 4.4)

All hyperparameters are set as the experimental setup in our paper. To reproduce the results, one needs to firstly install the environment and download datasets and models, both of which are available on Huggingface. Then, follow our guidelines to run the scripts.

## Step 1: Preparation

Prepare the environment, datasets, and models for the benchmark.

### Environment

(in the root / parent directory of this README file)

```
    conda create -n copymark python==3.10
    conda activate copymark
    pip install -r requirements.txt
```

### Datasets

All datasets are available on [huggingface](https://huggingface.co/datasets/CaradryanLiang/CopyMark). Download the `datasets.zip` to `datasets/` and unzip it. There are in total 10 datasets. Make sure the `datasets` directory has the following structure:

```
─diffusers           
│   └───datasets       
│           └───celeba-hq-2-5k-eval
│           └───celeba-hq-2-5k-test
│           └───coco2017-val-2-5k-eval
│           └───coco2017-val-2-5k-test
│           └───commoncatalog-2-5k-eval
│           └───commoncatalog-hq-2-5k-eval
│           └───ffhq-2-5k-eval
│           └───ffhq-2-5k-test
│           └───laion-aesthetic-2-5k-eval
│           └───laion-aesthetic-2-5k-eval
│           │   placeholder.txt
│   ...
```


### Models

We include three models for our benchmark: [Latent Diffusion Models (Celeba-HQ-256)](https://huggingface.co/CompVis/ldm-celebahq-256), [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), and [CommonCanvas XL C](https://huggingface.co/common-canvas/CommonCanvas-XL-C). Clone these models from huggingface to `models/` directory.


## Step 2: Reproducing benchmark results

Basically, the benchmark results consist of two parts: results on evaluation datasets (with appendix `eval` in the directory name) and results on test datasets (with appendix `test` in the directory name). Check our paper for the difference between these two parts. To reproduce these results:

- First, run baselines on evaluation datasets to get the first part of results as well as **result files** that include the thresholds calculated on the evaluation datasets. 

- Second, moving the result files to `experiments` directory because we need to test these thresholds on test datasets. 

- Third, run baselines on test datasets to get the second part of results.


### Part 1: Results on evaluation datasets

Run the following commands:
```
    scripts/exp_ldm_eval.sh
    scripts/exp_sd_eval.sh
    scripts/exp_sdxl_eval.sh
```

The result files will be placed in `outputs` directory, denoted by the name of the method and the model. Generally, the result files include:

```
    {method}_{model}_score_result.json      # main results, auc + tpr + thresholds
    {method}_{model}_image_log.json         # image filenames in the evaluation sequence
    {method}_{model}_running_time.json      # running time of the method on the model
    {method}_{model}_member_scores.pth      # scores of member data, used to visualizing score distributions
    {method}_{model}_nonmember_scores.pth   # scores of non-member data, used to visualizing score distributions
```

Specifically, for GSA we do not have member scores and non-member scores directly. Instead, we have the features (gradient norm) of member data and non-member data.

```
    {method}_{model}_features.npy      # features of member data & non-member data, used to visualizing score distributions
```

### Part 2: Results on test datasets

First, move the result files in the `outputs` directory to their corresponding subfolders in`experiments` directory. Make sure that `experiments` keep the following structure:
```
─diffusers                                     
│   └───experiments   
│           └───ldm
│               └───gsa_1
│               └───gsa_2
│               └───pfami
│               └───pia
│               └───secmi
│           └───sd
│               └───gsa_1
│               └───gsa_2
│               └───pfami
│               └───pia
│               └───secmi
│           └───sdxl
│               └───gsa_1
│               └───gsa_2
│               └───pfami
│               └───pia
│               └───secmi
```

**Note:** We leave the threshold files of our original experiments in `experiments` directory currently. If you only want to reproduce our results on test datasets, you can simply run the following commands.

Run the following commands:
```
    scripts/exp_ldm_test.sh
    scripts/exp_sd_test.sh
    scripts/exp_sdxl_test.sh
```

The result files will be also placed in `outputs` directory in a format similar to **Part 2**.

## Step 3: Score Distributions

After getting the thresholds by running baselins on evaluation datasets, score distributions can be visualized. To reproduce the result in the paper, run the following command:
```
    python utils/score_distribution.py
```
The visualization results will be show directly.

## Step 4: Case Studies

With the threshold files of GSA1 and GSA2 on Stable Diffusion v1.5, we can detect whether a batch of copyright images are included in the training dataset of Stable Diffusion v1.5, as we do in the case studies of the paper. To reproduce the result, run the following command:
```
    python scripts/run_case_studies.py
```
The visualization results will be show directly.
