# Set up with docker

This docker file describes a container that allows you to run the experiments on any Unix-based machine. GPU availability is recommended to train the models. 

# Requirements

- [install docker](https://docs.docker.com/install/)
- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

# Installation

## 1. Build the image

From this folder you can create the image `man:latest`

```sh
docker build -t man:latest docker/
```

## 2. Start the container

Start the container `man-gpu`

```sh
docker run --name man-gpu -it --rm --gpus all -v $(pwd):/workspace man:latest
```

This will start a bash session in the container.

## 3. Run an experiment

Run the following command for example:

```sh
./runDemo.sh
```
