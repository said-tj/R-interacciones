#!/usr/bin/env python
# coding: utf-8
# Codigo.
# In[85]:
# Importamos las librerias y dependencias necesarias.
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
# Los datos estan almacenados en un archivo .csv, en donde se especifica cada relacion que existe en el grupo que se esta analizando.

# In[86]:
datos = pd.read_csv('grupos.csv', index_col=None)

# In[87]:
datos.head(10)

# In[88]:
relationship_df = pd.DataFrame(datos)

# In[89]:
pd.set_option('display.max_rows', None)
relationship_df

# In[90]:
relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)
relationship_df

# In[91]:
relationship_df["value"] = 1
relationship_df = relationship_df.groupby(["source","target"], sort=False, as_index=False).sum()

# In[92]:
relationship_df.head(20)

# Red social (Grafo)

# In[93]:
# Create a graph from a pandas dataframe
G = nx.from_pandas_edgelist(relationship_df, 
                            source = "source", 
                            target = "target", 
                            edge_attr = "value", 
                            create_using = nx.Graph())

# In[94]:
plt.figure(figsize=(10,10))
pos = nx.kamada_kawai_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# In[95]:
from pyvis.network import Network
net = Network(notebook = True, width="1366px", height="768px", bgcolor='#222222', font_color='white')
node_degree = dict(G.degree)
#Setting up node size attribute
nx.set_node_attributes(G, node_degree, 'size')
net.from_nx(G)
net.show("rg.html")

# Visualizar comunidades dentro de la red

# In[96]:
# Importamos librerias necesarias para mostrar la comunidades en una red
import community as community_louvain
import community.community_louvain as community_louvain

# In[97]:
communities = community_louvain.best_partition(G)

# In[98]:
communities

# In[99]:
nx.set_node_attributes(G, communities, 'group')

# In[100]:
com_net = Network(notebook = True, width="1366px", height="768px", bgcolor='#222222', font_color='white')
com_net.from_nx(G)
com_net.show("rg_communities.html")

