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

* python
* torch
* matplotlib
* numpy
* pandas
* pillow
* pyyaml
* scikit-learn

## Set up with conda

### 1. Build the environment with conda: 
```
conda create -n MAN python=3.8.10 pyyaml=5.4.1 numpy=1.24.3 \
      matplotlib=3.7.1 scikit-learn=1.3.0 pandas=2.0.3 pillow=8.2.0 \
      pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Activate the environment:
```
conda activate MAN
```

### 3. Run the demo:
```
./runDemo.sh
```

## Set up with docker

This docker file describes a container that allows you to run the experiments on any Unix-based machine. GPU availability is recommended to train the models. 

### Requirements

- [install docker](https://docs.docker.com/install/)
- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

### Installation

#### 1. Build the image

From this folder you can create the image `man:latest`

```sh
docker build -t man:latest docker/
```

#### 2. Start the container

Start the container `man-gpu`

```sh
docker run --name man-gpu -it --rm --gpus all -v $(pwd):/workspace man:latest
```

This will start a bash session in the container.

#### 3. Run an experiment

Run the following command for example:

```sh
./runDemo.sh
```

## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file "./config/body/parameter(dataset=body,\
       task=link_pred,model=gcn_man).yaml"
```
The configuration files for each dataset are placed in a subdirectory of "config" directory. For instance, the configuration files of the "body" dataset are placed in the "config/body" subdirectory. Most of the parameters in the yaml configuration file are self-explanatory. 

Running the above command will output a log file in the corresponding subdirectory of "blog" directory. For instance, the log files of the "body" dataset are placed in the "blog/body" subdirectory. The log files record information about the experiment and validation metrics for the various epochs. Instead of manual analysis, one may utilize the "log\_analyzer.py" file to automatically extract the best performance in different evaluation criteria from the log file. For example:

```sh
python log_analyzer.py "./blog/body/dataset=body,task=link_pred,\
       model=gcn_man.log"
```

To execute the two commands together, you can directly run the "runDemo.sh" script for all the datasets.

```sh
./runDemo.sh
```

