import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nltk import FreqDist
import re, logging

if __name__ == '__main__':
   # Prevent the warnings from matplotlib:
   logging.getLogger('matplotlib.font_manager').disabled = True
   
   # Get processed data:
   print('Reading from CSV...', end=' ')
   relevant_data = pd.read_csv('processed/topTagTweets.csv', parse_dates=['publicationtime']).convert_dtypes()
   print('Done')
   
   # Construct a co-occurrence network
   G = nx.Graph()
   for hashtags in relevant_data['hashtags']:
      for i, hashtag1 in enumerate(hashtags):
         for hashtag2 in hashtags[i+1:]:
            G.add_edge(hashtag1.lower(), hashtag2.lower())  # Add edge between co-occurring hashtags

   # View graph, highlighting communities:
   communities = nx.community.greedy_modularity_communities(G)
   supergraph = nx.cycle_graph(len(communities))
   superpos = nx.spring_layout(G, scale=50, seed=429)
   centers = list(superpos.values())
   pos = {}
   for center, comm in zip(centers, communities):
      pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center, seed=1534))
   for nodes, clr in zip(communities, ('tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple')):
      nx.draw_networkx_nodes(G, pos=pos, node_color=clr, nodelist=nodes, node_size=50)
   nx.draw_networkx_edges(G, pos=pos)
   plt.tight_layout()
   plt.title('Hashtag Co-Occurrence Network')
   plt.savefig('img/hashtagsGraph.png')

   # Calculate centrality measures
   degree_centrality = nx.degree_centrality(G)
   betweenness_centrality = nx.betweenness_centrality(G, k=100)  # k is the number of random samples
   eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

   # Plot degree centrality
   plt.figure(figsize=(8, 6))
   plt.bar(degree_centrality.keys(), degree_centrality.values())
   plt.title('Degree Centrality')
   plt.xlabel('Hashtag')
   plt.ylabel('Centrality')
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig('img/degree_centrality.png')
   plt.close()

   # Plot betweenness centrality
   plt.figure(figsize=(8, 6))
   plt.bar(betweenness_centrality.keys(), betweenness_centrality.values())
   plt.title('Betweenness Centrality')
   plt.xlabel('Hashtag')
   plt.ylabel('Centrality')
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig('img/betweenness_centrality.png')
   plt.close()

   # Plot eigenvector centrality
   plt.figure(figsize=(8, 6))
   plt.bar(eigenvector_centrality.keys(), eigenvector_centrality.values())
   plt.title('Eigenvector Centrality')
   plt.xlabel('Hashtag')
   plt.ylabel('Centrality')
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig('img/eigenvector_centrality.png')
   plt.close()