# ADA-COVID

TensorFlow 2 implementation of **ADA-COVID: Adversarial Deep Domain Adaptation-Based
Diagnosis of COVID-19 from Lung CT Scans Using Triplet Embeddings**.

> Aria, M., Nourani, E., & Golzari Oskouei, A. (2022). ADA‐COVID: Adversarial Deep Domain Adaptation‐Based Diagnosis of COVID‐19 From Lung CT Scans Using Triplet Embeddings. *Computational Intelligence
> and Neuroscience*, 2022(1), 2564022.
> https://doi.org/10.1155/2022/2564022

<details><summary>Abstract</summary>Rapid diagnosis of COVID-19 with high reliability is essential in the early stages. To this end, recent research often uses medical imaging combined with machine vision methods to diagnose COVID-19. However, the scarcity of medical images and the inherent differences in existing datasets that arise from different medical imaging tools, methods, and specialists may affect the generalization of machine learning-based methods. Also, most of these methods are trained and tested on the same dataset, reducing the generalizability and causing low reliability of the obtained model in real-world applications. This paper introduces an adversarial deep domain adaptation-based approach for diagnosing COVID-19 from lung CT scan images, termed ADA-COVID. Domain adaptation-based training process receives multiple datasets with different input domains to generate domain-invariant representations for medical images. Also, due to the excessive structural similarity of medical images compared to other image data in machine vision tasks, we use the triplet loss function to generate similar representations for samples of the same class (infected cases). The performance of ADA-COVID is evaluated and compared with other state-of-the-art COVID-19 diagnosis algorithms. The obtained results indicate that ADA-COVID achieves classification improvements of at least 3%, 20%, 20%, and 11% in accuracy, precision, recall, and F1 score, respectively, compared to the best results of competitors, even without directly training on the same data.</details>

## Architecture

![ADA-COVID Approach](https://raw.githubusercontent.com/MehradAria/ADA-COVID/main/Fig%201%20-%20ADACOVID.png)

- **Feature extractor**: pretrained ResNet50 (ImageNet), 224x224 input
- **Classifier head**: Dense-BN-ReLU-Dropout (x2) -> 64-dim L2-normalized embedding, trained with triplet loss
- **Discriminator head**: Gradient Reversal Layer -> Dense-BN-ReLU-Dropout (x2) -> Sigmoid (source/target)
- **Loss**: `lambda_y * TripletLoss + lambda_d * BinaryCrossEntropy`, with `lambda_y=4`, `lambda_d=1`
- **Test phase**: discriminator removed, 2-neuron softmax head added on top of the classifier

## Project structure

```
ada_covid/
    __init__.py           Package metadata
    config.py             Hyperparameters and default configuration
    utils.py              Seeding and environment info
    data.py               Data loading, normalization, augmentation, batching
    layers.py             Gradient Reversal Layer
    losses.py             Semi-hard triplet loss
    models.py             Model architecture and compilation
    training.py           Adversarial training loop and Stage 2 fine-tuning
    evaluation.py         Metrics and single-image inference
    visualization.py      Training curve plotting
main.py                   End-to-end pipeline entry point
requirements.txt
setup.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset preparation

Two plain-text files are required, each listing one image path and label per line:

```
/path/to/dataset/COVID/ct_scan_001.png 1
/path/to/dataset/non-COVID/ct_scan_100.png 0
```

Label `1` = COVID-19 positive, label `0` = non-COVID-19.

Datasets used in the paper:

- **Source**: [SARS-CoV-2 CT scan dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) (2482 images)
- **Target**: [COVID-19 CT dataset](https://github.com/UCSD-AI4H/COVID-CT) (746 images)
- **Benchmark**: [COVID-19 Lung CT Scans](https://www.kaggle.com/mehradaria/covid19-lung-ct-scans) (8439 images)

```
Aria, M., Ghaderzadeh, M., Asadi, F., & Jafari, R. (2021). COVID-19 lung CT scans: A large dataset of lung CT scans for COVID-19 (SARS-CoV-2) detection (Kaggle) [Dataset]. https://doi.org/10.34740/kaggle/dsv/1875670
```

Use `ada_covid.data.create_dataset_txt` to auto-generate these files from a directory
structure of `COVID/` and `non-COVID/` subfolders.

## Usage

```shell
python main.py
```

This runs Stage 1 adversarial domain-adaptation training, Stage 2 softmax head
fine-tuning with 5-fold cross-validation, evaluation on source and target sets, and
saves the trained models plus a training-progress plot.

## Citation
Please cite the following Open Access [paper](https://doi.org/10.1155/2022/2564022):

```bibtex
@article{aria2022adacovid,
  title   = {ADA-COVID: Adversarial Deep Domain Adaptation-Based Diagnosis of COVID-19 from Lung CT Scans Using Triplet Embeddings},
  author  = {Aria, Mehrad and Nourani, Esmaeil and Golzari Oskouei, Amin},
  journal = {Computational Intelligence and Neuroscience},
  volume  = {2022},
  pages   = {2564022},
  year    = {2022},
  doi     = {10.1155/2022/2564022}
}
```
```APA
Aria, M., Nourani, E., & Golzari Oskouei, A. (2022). ADA-COVID: Adversarial deep domain adaptation-based diagnosis of COVID-19 from lung CT scans using triplet embeddings. Computational Intelligence and Neuroscience, 2022, 2564022. https://doi.org/10.1155/2022/2564022
```
