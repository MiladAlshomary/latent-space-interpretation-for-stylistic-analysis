import pandas as pd
import json
import sklearn
import glob
import pickle as pkl
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
import numpy as np
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, '../../style_generation_pipeline')

from data import *
from cluster_representation import *

interp_space_path = '../datasets/hiatus_data/interp_space_148_clusters/interpretable_space.pkl'
interp_space_rep_path = '../datasets/hiatus_data/interp_space_148_clusters/interpretable_space_representations.json'
style_feat_clm = 'llm_tfidf_rep'
#style_feat_clm = 'llm_con_rep'

#'../data/explainability/clusterd_authors_with_style_description.pkl'
interpretable_space = pkl.load(open(interp_space_path, 'rb'))
del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}

#Load interp space representations
interpretable_space_rep_df = pd.read_json(interp_space_rep_path)
dimension_to_style  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}


X = np.array([x[1] for x in dimension_to_latent.items()])
y = np.array([x[0] for x in dimension_to_latent.items()])

X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
X_embedded.shape

x = X_embedded[:,0]
y = X_embedded[:,1]

names = np.array([" \n ".join(dimension_to_style[item[0]]) for item in dimension_to_latent.items()])

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y, s=100, cmap=cmap, norm=norm)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)
    

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()