# Statistical-autoencoder

This is an implementation of : Efficient classification using the latent space of a Non-Parametric Supervised Autoencoder for metabolomics datasets of clinical studies. In this repository, you will find the code to perform the clinical study described in the paper. For the statistical and comparison part, you can find it here : https://github.com/Gustoaxel/Clinical-autoencoder
  
When using this code , please cite Barlaud, M., Guyard, F.: Learning sparse deep neural networks 
using efficient structured projections on convex constraints for green ai. ICPR 2020 Milan Italy (2020)

and 

Axel Gustovic, Celine Ocelli, Thierry Pourcher and Michel Barlaud : Efficient diagnostic using the 
latent space ofa Non-Parametric Supervised Autoencoderfor metabolomics datasets


## Table of Contents
***
1. [Installation](#installation)
2. [How to use](#use)
  
    
## Installation : 
***

First, you will need a python runtime environment. If you don't have one on your computer we recommend you to download anaconda (https://www.anaconda.com/products/individual#Downloads). It is a platform that brings together several IDEs for machine learning. In the rest of this tutorial we will use Spyder. 
You can now download the code in zip format and unzip it on your computer.
Then, to execute our script, we will need several dependencies. To install them you will have to run this command in the spyder console (at the bottom right).
```
$ conda install -c anaconda pip
$ cd path/to/project
$ pip install -r requirements.txt (Warning, before launching this command you must go to the directory where the requirements.txt is located)
```

To install captum make sure you have a c++ compiler

## How to use : 

Everything is ready, now you have to open the code in spyder (top left button). 
Then run it with the Run files button. It is possible to change the parameters and the database studied directly in the code. 

Note that we have provided in this directory only the LUNG database. To obtain the Brain and covid databases please contact Barlaud Michel (barlaud@i3s.unice.fr) or Pourcher Thierry (thierry.pourcher@univ-cotedazur.fr)

Here is a list of modifiable parameters with our values : 

| Parameters | line in code | recommended value |
|:--------------|:-------------:|--------------:|
| ETA | 129 | 75|
| Seed | 59 | 4, 5, 6 |
| Database | 85 | Lung |
| Projection | 124 | l11 |
| Scaling | 145 | True |
