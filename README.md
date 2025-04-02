# Deep-EM Playground: Bringing Deep Learning to Electron Microscopy Labs

This codebase is part of the Deep-EM Playground. For more details, please see our [webpage](https://viscom-ulm.github.io/DeepEM/).


## Image to Value(s)


### Primary Focus: Explainable Object Counting in Microscopy Images
### Application: Explainable Virus Capsid Quantification
#### Challenge: Deep Learning as Black Box
#### Required Labels: Location Labels


TL;DR 🧬✨ We developed a regression model to quantify virus capsids and their mutation stages ("naked", "budding", "enveloped") during secondary envelopment in TEM images. Researchers can adapt the provided notebook within the primary focus area for their own EM data analysis (i.e. counting midrochondia).

![Teaser](./images/Teaser.png)

## Setup

### Lightning AI
<a target="_blank"
href="https://lightning.ai/hannah-kniesel/studios/deepem-explainable-object-counting-in-microscopy-images">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg"
    alt="Open In Studio" />
</a>

Start immediately using the Lightning AI Studio template by clicking the button above - no additional setup required. This will dublicate the studio in your own teamspace, allowing you to experiment with the code base.

### Local Setup 
For a quick setup, we offer the use of `conda`, `pip` or `docker`. This will provide all needed libraries as well as common libraries used for deep learning (for more details you can check `requirements.txt`). Of course you are free to install additional dependencies, if needed.  

#### Conda (LightningAI)
On your machine, run:
```bash
conda env create -f environment.yml
conda activate deepem_objectcounting
```

If you are working on [LightingAI](https://lightning.ai/) Studios, there will be a base environment, which you can update with the needed dependencies, by running: 
```bash
conda env update --file environment.yml --prune
```

#### Pip
When working with `pip`, please make sure you have a compatible python version installed. This use case was tested on `python == 3.12.5` with `cuda==12.1` and `cudnn9`.
Next, you can run
```bash
pip install -r requirements.txt
```

#### Docker
Build your own image with: 
```bash 
docker build -t deepem_objectcounting .
```
This will generate a docker image called `deepem_objectcounting`. 

or use the existing docker image from `hannahkniesel/deepem_objectcounting`. 

Start the docker with this command: 
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /local_dir/:/workspace/ --name <container-name> <image-name> bash
```
For example the full command could look like this: 
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v "/Documents/DeepEM-Explainable-Object-Counting/":/workspace/ --name deepem hannahkniesel/deepem_objectcounting bash 
```

Inside the container start `jupyter notebook`
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
Click on the link printed in the console to open the jupyter notebook in the browser.



## Citation

If you find this code useful, please cite us: 

    @inproceedings{
    }