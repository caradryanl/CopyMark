Benchmark
===================


# Reproduction

## Preparation

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


## Evaluation results

### Running on Evaluation Datasets


### Running on Test Datasets


## Score Distributions

## Case Studies

