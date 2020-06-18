---
layout: post
mathjax: true
title: Analyzing Amazon Reviews
---

## Analyzing Amazon Reviews

#### How to employ Data Science to determine the authenticity of amazon reviews.

In physics, the mass-energy equivalence is stated 
by the equation $E=mc^2$, discovered in 1905 by Albert Einstein.

#### Here is some helpful code


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
```


```python
# tSNE with two components
X = df_for_clustering
y = df_for_clustering['star_rating']

# We need to scale the data since tSNE is also distance based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

my_tSNE = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

X_tSNE = my_tSNE.fit_transform(X_scaled)
```
