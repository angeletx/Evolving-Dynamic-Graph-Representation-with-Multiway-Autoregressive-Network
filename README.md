MAN
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

For new data sets, please place them in the 'data' folder.

## Requirements
* python==3.8.10
* matplotlib==3.7.1
* numpy==1.24.3
* pandas==2.0.3
* pillow==8.2.0
* pyyaml==5.4.1
* scikit-learn==0.24.2
* torch==2.0.0


## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file "./config/body/parameter(dataset=body,
task=link_pred,model=gcn_man).yaml"
```
The configuration files for each dataset are placed in a subdirectory of "config" directory. For instance, the configuration files of the ``body'' dataset are placed in the "config/body" subdirectory. Most of the parameters in the yaml configuration file are self-explanatory. 

Running the above command will output a log file in the corresponding subdirectory of "blog" directory. For instance, the log files of the "body" dataset are placed in the "blog/body" subdirectory. The log files record information about the experiment and validation metrics for the various epochs. Instead of manual analysis, one may utilize the "log\_analyzer.py" file to automatically extract the best performance in different evaluation criteria from the log file. For example:

```sh
python log_analyzer.py "./blog/body/dataset=body,task=link_pred,model=gcn_man.log"
```

To execute the two commands together, you can directly run the "runDemo.sh" script for all the datasets.

```sh
./runDemo.sh
```

