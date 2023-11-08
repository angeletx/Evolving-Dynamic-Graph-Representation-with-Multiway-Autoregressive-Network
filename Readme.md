MAN
=====

This repository contains the code for [Multiway Autoregressive Network].

## Requirements
  * PyTorch 1.8.1 
  * Python 3.8

## Usage

see runDemo.sh

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file "./config/email_eu_dept1/parameter(dataset=email_eu_dept1,task=link_pred,model=egcn_man,learningRate=0.001000,historyStep=5,gcnHiddenFeature=[64x20]).yaml"
```
Most of the parameters in the yaml configuration file are self-explanatory. 

The file could be manually analyzed, alternatively 'log_analyzer.py' can be used to automatically parse a log file and to retrieve the evaluation metrics at the best validation epoch. For example:

see runDemo.sh

```sh
python log_analyzer.py "./blog/email_eu_dept1/dataset=email_eu_dept1,task=link_pred,model=egcn_man,learningRate=0.001000,historyStep=5,gcnHiddenFeature=[64x20].log"
```

