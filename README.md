[![GitHub license](https://img.shields.io/github/license/hamelsmu/code_search.svg)]()
[![Python 3.9.6](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-396/)

# *MicNet Toolbox: visualizing and deconstructing a microbial network* 

## Introduction

This project contains a set of tools to detect, analyze and visualize microbial interaction networks from compositional data. The main objective was to implement different tools of the different algorithms to reconstruct, analyze and visualize a co-ocurrence network with microbial interactions. For that, we use standard algorithms for compositional data processing (SparCC), Network or Graphics Algorithms and Umap Algorithms.


This project is part of the collaborative research between Ixulabs and the Laboratory of Experimental and Molecular Evolution, Institute of Ecology, UNAM.

* [Laboratory of Experimental and Molecular Evolution, Instituto de Ecología, UNAM](http://web2.ecologia.unam.mx/perfiles/perfil.php?ID=1237852985093)

* [Ixulabs](https://ixulabs.com/)


## Project Overview
---

The project is divided into three parts:
* Visualization
* SparCC
* Network Algorithms

**Note:** Each package has a REAME.md file with a specific description of how the code works. 

### [Visualization](https://umap-learn.readthedocs.io/en/latest/clustering.html)

Compositional data cannot be processed or worked like normal data, it has particular geometric and statistical properties. Using different transformations (Dirichlet transformation, Normalization or CLR) and the original data we estimate a mapping through the UMAP algorithm to the Hyperbolic Space. This to have a visualization of the data and its possible interactions. Furthermore, through the HDBSCAN algorithm we detect clusters based on density and which points are possible outliers and noise.

### [SparCC](sparcc/README.md)
 
 This algorithm is standard for estimating correlation values from compositional data, in order to infer the network's connections. We made several modifications to the 
 [original version of the algorithm
 ](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687) to improve the ability to process large samples of data. Part of the stages were parallelized and the execution of the algorithm and its statistical tests were better controlled. To run this on your local please refer to the README.md in the sparcc folder.

### [Network Algorithms](network_alg/README.md)

Network analyses were used to characterise both the overall structure and the local interactions of the microbial network, in which each OTU was represented as a node and the correlations found by SparCC as undirected weighted edges, such that an edge between two nodes implies a relationship between the two corresponding OTUs. Given that most network analyses can only handle positive interactions, we normalized the SparCC correlation matrix from -1 to 1 to a range from 0 to 1, except for the structural balance analysis which directly uses the positive and negative correlation values. The dashboard includes the calculation of large scale metrics of the network, structural balance analysis and community/HDBSCAN subnetwork analysis. To run percolation analysis and topology comparison please refer to the README.md in the network_alg folder.


## Local use
If you do not wish to use the web app of MicNet which has limited capacity, and you would prefer using **your own computer disk and RAM resources**, all you have to do is follow these steps. To do this, you must first have the *conda* package manager installed. If you do not have it installed, we recommend that you install the miniconda or anaconda packages. You can find the instructions to install it at this link:

[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

[Anaconda](https://www.anaconda.com/products/individual)

Once you have conda working on your computer the process is the following:

1. First be sure to be situated in the MicNet repository folder that you have clone into your computer. Then, the first step to run the dashboard is to create the environment, thus, first you must run the following code in the command line or terminal:

    ~~~bash
    conda env create -f environment.yml 
    ~~~

    Be sure to check if the environment was created by typing in your command line the following:

    ~~~bash
    conda env list 
    ~~~

    You should see MicNet-env listed among your environments.

2. If the MicNet-env environment is in the list, you then need to activate it using the following code:

    ~~~bash
    conda activate  MicNet-env
    ~~~

3. Finally, you just need to run the following code to get the web app running, NOTE that this app will be using your computer's computational resources (disk and RAM), thus giving you more power than the one we provide with free memory resources [here](http://micnetapplb-1212130533.us-east-1.elb.amazonaws.com). 
    
    ~~~bash
    streamlit run app.py
    ~~~

    The application will open in your browser at the port: localhost:8501


## Data Details

As validation data, we use the Kombucha data set described in [Arikan et al., (2020)](https://onlinelibrary.wiley.com/doi/full/10.1111/1750-3841.14992). .All data can be find in the folder named "Data". The "Kombucha_abundance_table.txt" can be used as input to run the UMAP/HDBSCAN and SparCC modules. We have also included the co-occurence matrix ("Kombucha_Sparcc_matrix.csv") and the HDBSCAN output file ("Kombucha_HDBSCAN.csv") which can be used as input to test the network module. The kombucha example can be easily inspected using the web dashboard at [MicNet dashboard](http://micnetapplb-1212130533.us-east-1.elb.amazonaws.com).

As a case study we used the Domos Archean data set described in [Espinosa-Asuar et al. 2021](https://www.biorxiv.org/content/10.1101/2021.03.04.433984v1.full), a database with more than 2,000 OTUs. All data can be find in the folder named "Data". The "Domos_abundance_table.txt" can be use as input for the UMAP/HDBSCAN and Sparcc modules. We have also provided the co-ocurrence matrix in the file "Domos_Sparcc_matrix.csv" and the HDBSCAN output file ("Domos_HDBSCAN.csv") which can be input to test the network module. Given the size of the Domos dataset, to run this example you should run the dashboard locally in your computer, it cannot be run in the web version.

## Licenses
[MIT License](LICENSE).
