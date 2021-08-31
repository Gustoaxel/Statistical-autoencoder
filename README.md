# Statistical-autoencoder

This is an implementation of : An efficient diagnostic that uses the latent space of a Non-Parametric Supervised Autoencoder for metabolomic datasets of clinical studies. In this repository, you will find the code to perform the statistical study described in the paper. For the clinical part only, you can find it here : https://github.com/Gustoaxel/Clinical-autoencoder
  
When using this code , please cite Barlaud, M., Guyard, F.: Learning sparse deep neural networks 
using efficient structured projections on convex constraints for green ai. ICPR 2020 Milan Italy (2020)

and 

Axel Gustovic, Celine Ocelli, Thierry Pourcher and Michel Barlaud : An efficient diagnostic that uses the latent space of a Non-Parametric Supervised Autoencoder for metabolomic datasets of clinical studies


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
$ pip install -r requirements.txt 
```
(Warning, before launching this command you must go to the directory where the requirements.txt is located)

To install pytorch make sure you have a c++ compiler. If this is not the case, please follow the instructions in this link: https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-160 for windows or install Xcode app if you use mac os

## How to use : 

Everything is ready, now you have to open the code in spyder (top left button). In our case the files are named : "script_diagnostic.py" and "Script_autoencoder_4folds.py"

Then run it with the Run files button. It is possible to change the parameters and the database studied directly in the code. 

Note that we have provided in this directory only the LUNG and Brain databases. To obtain the Covid database please contact Barlaud Michel (barlaud@i3s.unice.fr) or Pourcher Thierry (thierry.pourcher@univ-cotedazur.fr)

To use your own data it is important that it is in the same format as the provided databases. If this is the case then you just have to put the .csv file in the datas/FAIR directory and then modify the file_name variable = "something.csv" in the code. 

Here is a list of modifiable parameters with our values for "Script_autoencoder_4folds.py": 

| Parameters | line in code | recommended value |
|:--------------|:-------------:|--------------:|
| ETA | 81 | 75|
| Seed | 80 | 4, 5, 6 |
| Database | 106 | Lung |
| Projection | 145 | l11 |
| Metabolomic selection | 156 | True |
| Scaling | 163 | True |

Here is a list of modifiable parameters with our values for "script_diagnostic.py": 

| Parameters | line in code | recommended value |
|:--------------|:-------------:|--------------:|
| ETA | 85 | 600 |
| Seed | 84 | 5 |
| Database | 111 | Lung |
| Projection | 150 | l11 |
| Metabolomic selection | 161 | False |
| Scaling | 167 | True |

