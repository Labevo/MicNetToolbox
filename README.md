[![GitHub license](https://img.shields.io/github/license/hamelsmu/code_search.svg)]()
[![Python 3.9.6](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-396/)

# *MicNet Toolbox: Deconstructing a microbial interaction network* 

## Introduction

This project contains a set of tools to detect, analyze and visualize microbial interaction networks from composition data. The main objective was to implement different tools of the different algorithms to reconstruct the possible interactions between the microbial interactions. For that, we use standard algorithms for compositional data processing (SparCC), Network or Graphics Algorithms and Umap Algorithms.

This project is part of the collaborative research between Ixulabs and the Laboratory of Experimental and Molecular Evolution, Institute of Ecology, UNAM.

* [Laboratory of Experimental and Molecular Evolution, Instituto de Ecología, UNAM](http://web2.ecologia.unam.mx/perfiles/perfil.php?ID=1237852985093)

* [Ixulabs](https://ixulabs.com/)


## Project Overview
---

The project is divided into three parts:

* SparCC
* Network Algorithms
* Visualization

### [SparCC](SparCC/README.md)
 
 This algorithm is standard for estimating correlation values from composition data, in order to infer the network connection. We made several modifications to the 
 [original version of the algorithm
 ](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687), the first of which is to improve the ability to process large samples of data. Part of the stages were parallelized and the execution of the algorithm and its statistical tests were better controlled.



### [Network Algorithms](SNA/README.md)

Network analysis was used to characterise both the overall structure and the local interactions of the microbial network, in which each OTU was represented as a node and the correlations found by SparCC as undirected weighted edges, such that an edge between two nodes implies a relationship between the two corresponding OTUs. Given that most network analyses can only handle positive interactions, we normalized the SparCC correlation matrix from -1 to 1 to a range from 0 to 1, except for the structural balance analysis which directly uses the positive and negative correlation values.

### [Visualization](Visualization/README.md)

Composition data cannot be processed or worked like normal data, it has particular geometric and statistical properties. Using different transformations (Dirichlet transformation, Normalization or CLR) and the original data to apply some metrics to data processing and we estimate a mapping through the UMAP algorithm to the Hyperbolic Space. This to have a visualization of the data and its possible interactions. On the other hand, through the HDBSCAN algorithm we detect from the embedding which points are possible outliers.

**Note:** Each package has a REAME.md file with a specific description of how the code works. 

## Setup 

To configure the environment, you must first have the *conda* package manager installed. If you do not have it installed, we recommend that you install the miniconda packages. You can find the instructions to install it at this link:

[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

To create the environment, you must run the following code in the folder of this repository.

~~~bash
conda env create -f environment.yml 
~~~

Check if the environment was created:
~~~bash
conda env list 
~~~

If the MicNet-env environment is in the list, you can activate it using the following code:

~~~bash
conda activate  MicNet-env
~~~

## Local use
If you want to use it locally, it must be inside the MicNetTools folder and run the following code:
~~~bash

streamlit run app.py

~~~
The application will open in your browser at the port: http://localhost:8501 

**Note: note: remember to activate the MicNet-env environment first**

## Data Details

As validation data, we use the Kombucha data set described in [Arikan et al., (2020)](https://onlinelibrary.wiley.com/doi/full/10.1111/1750-3841.14992) 

## Licenses
[MIT License](LICENSE).
