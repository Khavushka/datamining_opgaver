import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plot_tree(clf, 
          feature_names=iris.get('feature_names'), 
          class_names=iris.get('target_names'))

plt.show()