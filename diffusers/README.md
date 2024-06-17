Benchmark
===================


# Reproduction

## Preparation

Prepare the environment, datasets, and models for the benchmark.
### Environment

(in the root / parent directory of this README file)

```
    conda create -n copymark python==3.10
    pip install -r requirements.txt
    conda activate copymark
```

### Datasets

All datasets are available on [huggingface](https://huggingface.co/datasets/CaradryanLiang/CopyMark). Download the `datasets.zip` to `datasets/` and unzip it. There are in total 10 datasets. Make sure the `datasets` directory has the following structure:

```
─diffusers           # benchmark on diffusers
│   └───datasets        # data: put the datasets here
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


## Evaluation results

Basically, the evaluation results consist of two parts: results on evaluation datasets (with appendix `eval` in the directory name) and results on test datasets (with appendix `test` in the directory name). To reproduce the results:

- First, run baselines on evaluation datasets to get the first part of results as well as **the thresholds of different baselines**. 

- Second, moving the thresholds to `experiments` directory because we need to test these thresholds on test datasets. 

- Third, run baselines on test datasets to get the second part of results.


### Step 1: Running baselines on evaluation datasets

Run the following commands:
```
    scripts/exp_ldm_eval.sh
    scripts/exp_sd_eval.sh
    scripts/exp_sdxl_eval.sh
```

### Step 2: Moving the thresholds to `experiments` directory

Move the results on evaluation datasets in the `outputs` directory to their corresponding subfolders in`experiments` directory. Make sure that `experiments` keep the following structure:
```
─diffusers                                      # benchmark on diffusers
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

### Step 3: Running baselines on test datasets

Run the following commands:
```
    scripts/exp_ldm_test.sh
    scripts/exp_sd_test.sh
    scripts/exp_sdxl_test.sh
```

## Score Distributions

After getting the thresholds by running baselins on evaluation datasets, we can then visualize the score distribution by running the following command:
```
    utils/score_distribution.py
```

## Case Studies
With the thresholds of GSA on Stable Diffusion v1.5, we can predict the membership of copyright images as the case study in the paper. Run the following command:
```
    python scripts/run_cases.py
```
