# TDPR: Prioritize Test Inputs for DNNs Using Training Dynamic
## Structure
- models: implementations of DNNs studied in TDPR
- data: the file used to save nomianl datasets and corrupted datasets
- results: experimental data including results of preliminary study and experiments
## Usage
We prepare a demo for TDPR, which uses results on corrupted test data (type=gaussian_noise) from Cifar-10-C.

`python demo.py`

If you want to run our code, please download the corrsponding datasets and run following code:

`python TDPR.py`

## Corrupted Data
- CIFAR-10-C: https://zenodo.org/records/2535967
- Tiny-ImageNet-C: https://zenodo.org/records/2536630
- IMDB and SMS are corrupted by Corrupted-Text： https://github.com/testingautomated-usi/corrupted-text
## Nominal Data
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Tiny_ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- IMDB: https://ai.stanford.edu/~amaas/data/sentiment/
- SMS: https://archive.ics.uci.edu/dataset/228/sms+spam+collection