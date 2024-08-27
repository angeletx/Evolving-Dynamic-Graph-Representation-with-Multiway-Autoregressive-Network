GCN MAN
=====

This his repository contains the code for [Evolving Dynamic Graph Representations with Multiway Autoregressive Network for Temporal Link Prediction].

## Data

6 datasets were used in the paper:

- 16-body problem: See the 'data/body' folder. 
- Enron: https://cs.cmu.edu/∼enron/
- Email Dept1: https://snap.stanford.edu/data/email-Eu-core-temporal.html
- College: https://snap.stanford.edu/data/CollegeMsg.html
- Autonomous Systems: https://snap.stanford.edu/data/as-733.html
- Epinion Trust: https://cse.msu.edu/∼tangjili/trust.html

For downloaded data sets please place them in the 'data' folder.

## Requirements
  * PyTorch 1.0 or higher
  * Python 3.8

## Set up with Docker

This docker file describes a container that allows you to run the experiments on any Unix-based machine. GPU availability is recommended to train the models. Otherwise, set the use_cuda flag in parameters.yaml to false.

### Requirements

- [install docker](https://docs.docker.com/install/)
- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)


## Requirements
  * PyTorch 1.8.1 
  * Python 3.8
  * cycler==0.10.0
  * joblib==1.0.1
  * kiwisolver==1.3.1
  * matplotlib==3.4.2
  * numpy==1.20.2
  * pandas==1.2.5
  * Pillow==8.2.0
  * pyparsing==2.4.7
  * python-dateutil==2.8.1
  * pytz==2021.1
  * PyYAML==5.4.1
  * scikit-learn==0.24.2
  * scipy==1.6.2
  * six==1.16.0
  * threadpoolctl==2.1.0
  * torch==1.8.1+cu111
  * typing-extensions==3.10.0.0
  * scikit-learn=0.24.2
  * yaml=0.2.5

## Usage

see runDemo.sh

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file "./config/body/parameter(dataset=body,task=link_pred,model=gcn_man,learningRate=0.001000,historyStep=5).yaml"
```
Most of the parameters in the yaml configuration file are self-explanatory. 

The file could be manually analyzed, alternatively 'log_analyzer.py' can be used to automatically parse a log file and to retrieve the evaluation metrics at the best validation epoch. For example:

see runDemo.sh

```sh
python log_analyzer.py "./blog/body/dataset=body,task=link_pred,model=gcn_man,learningRate=0.001000,historyStep=5.log"
```

