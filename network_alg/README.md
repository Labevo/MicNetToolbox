
# *Network analysis of co-ocrrence network obtained from SparCC* 

## Introduction

This module is to run the following network analysis that are not present in the web dashboard:
* Percolation analysis
* Network topology 
* Degree distribution comparison

The other analyses presented in Favila, Madrigal-Trejo et al. (in revision) are already integrated in the web dashboard.


## Percolation analysis
---
The percolation analysis consists of removing nodes and their corresponding edges and analyzing how much the network's properties are disrupted. The percolation simulation consists of n iterations; in each iteration a percentage of the nodes (with default value of 0.1, but this can be specified by the user) is removed along with all of their edges.
We have provided two percolations functions to run percolation either by a type of centrality or by defined groups (such as clusters or taxa groups). To import these funcions locate yourself in the MicNetToolbox folder and run the following code:

~~~python
from network_alg import percolation_sim, percolation_by_group
~~~

To use these two functions you will need to input a correlation matrix (note that it should be a square matrix). The suggested usage is as follow:

~~~python
import pandas as pd

#Read correlation file as a pandas dataframe 
corr = pd.read_csv('correlation_file.csv')

#Run percolation removing nodes by degree centrality
percolation = percolation_sim(corr, prem =0.1, per_type='deg_centrality')

#Run percolation removing nodes by group
percolation = percolation_by_group(corr, prem=0.1, groups=grouplist)
~~~

Note than when using percolation_sim, the removal of nodes can be done randomly, by degree centrality, closeness centrality or betweeness centrality by changing the per_type parameter to 'random', 'deg_centrality', 'clos_centrality' and 'bet_centrality', respectively.

In the case of running a percolation where groups will be removed a list containing group id for each OTU nees to be provided. The length of the list needs to be the same as the number of rows in the corr matrix.

## Network topology
---

MicNet includes the computation of the distribution of several of this large-scale metrics under the assumption that the underlying topology is: 1) a random Erdos-Renyi network, built using function nx.erdos_renyi_graph, 2) a small world Watts-Strogatz built using nx.watts_strogatz_graph function, or 3) a scale-free Barabási-Albert network built using nx.barabasi_albert_graph function. 

To import the function locate yourself in the MicNetToolbox folder and run the following code:

~~~python
from network_alg import topology_boostrap
import pandas as pd

#Read correlation file as a pandas dataframe 
corr = pd.read_csv('correlation_file.csv')

#Run boostrap
df_rand, df_small, df_scale = topology_boostrap(corr, n_boot=100)
~~~

The topology_boostrap function takes a correlation matrix as input and returns three dataframes with the distribution for several large-scale metrics under the assumption that the network with the same density and average degree but with a defined topology, either random, small-world or scale-free. The number of simulated networks can be controlled with parameter n_boot.


We suggest plotting the resulting values of to inspect each metric as follow:

~~~python
#Plotting average shortest path under the asumption of a random, small-world and scale-free topology.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.kdeplot(df_small['Average shortest path'], color='g', shade=True)
sns.kdeplot(df_rand['Average shortest path'], color='r', shade=True)
sns.kdeplot(df_scale['Average shortest path'], color='b', shade=True)
~~~

For a more detailed description of topology comparison refer to our paper Favila, Madrigral-Trejo et al (in revision).

## Degree comparison
---

Degree distributions can also be used to discriminate between network topologies. Thus, we have included in the MicNet toolbox a function that plots the Complementary Cumulative Distribution Function (CCDF) of the degrees of the given network and compares it with the CCDF of a simulated comparable random, scale-free and small-word network on a log-log scale. 

To import the function locate yourself in the MicNetToolbox folder and run the following code:

~~~python
from network_alg import degree_comparison
import pandas as pd

#Read correlation file as a pandas dataframe 
corr = pd.read_csv('correlation_file.csv')

#Run function
CCDF = degree_comparison(corr, topology ='random', bins=20)
~~~

the degree_comparison function returns a dataframe with the bins and CCDF for the corr matrix, and the bins and CCDF for an equivalent network with defined topology (which can be specified in the topology parameter as 'random', 'small_world' or scale_free).

We suggest to plot the results as follows:

~~~python
#Plotting average shortest path under the asumption of a random, small-world and scale-free topology.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7.5,6))
sns.lineplot(x=df.Data_bins,y=df.Data_CCDF, color = 'black',lw= 3)
sns.lineplot(x=df.Simulated_bins,y=df.Simulated_CCDF, color = 'r',lw= 3)
plt.xlabel('Degree')
plt.ylabel('CCDF')

~~~