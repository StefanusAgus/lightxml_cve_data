# LightXML for CVE Dataset

Adapted from the paper "LightXML: Transformer with dynamic negative sampling for High-Performance Extreme Multi-label Text Classiﬁcation"

## Requirements

Install Pytorch (Follow https://pytorch.org/)
``` bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Install other requirements
``` bash
pip install -r requirements.txt
```

Please also install apex as follows
``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
or if the above command failed, use:
``` bash
cd apex
python setup.py install
```

## Datasets
As inputs, the data preparation script expects three files:
1. Train CSV file with sparse label format
2. Test CSV file with sparse label format
3. CVE-Labels CSV file

For examples of the three input data, please refer to the "dataset_train.csv", "dataset_test.csv", and "cve_labels_merged_cleaned.csv" in the dataset folder.

## Run Commands
All of the following commands are run from the base folder of LightXML.

Dataset preparation
``` bash
mkdir dataset/splitted
mkdir dataset/cve_data
python dataset_preparation.py --training_csv=dataset/dataset_train.csv --test_csv=dataset/dataset_train.csv --cve_labels_csv=dataset/cve_labels_merged_cleaned.csv
```
The above command will generate the dataset in the format expected by lightxml in the dataset/cve_data folder. This dataset will be utilized in the LightXML training and testing.

Model training and fine-tuning
BERT
``` bash
python src/main.py --lr 1e-4 --epoch 20 --dataset cve_data --swa --swa_warmup 10 --swa_step 200 --batch 16
```

RoBERTa
``` bash
python src/main.py --lr 1e-4 --epoch 20 --dataset cve_data --swa --swa_warmup 10 --swa_step 200 --batch 16  --bert roberta
```

XLNet
``` bash
python src/main.py --lr 1e-4 --epoch 20 --dataset cve_data --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet
```

Model Evaluation
``` bash
python src/ensemble.py --dataset cve_data
```