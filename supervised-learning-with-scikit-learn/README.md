https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
# __Supervised Learning with Scikit-learn__
<br>


## __Classification__
---

### __General Notes__
- Features = predictive variables = indepentent variables
- Target variable = response variable = dependent variable
- Train/test split
    - test_size (defaults to 25%)
    - random_state - sets the random seed to reproduce results
    - stratify - achieves same distribution or proportion of various targets in train/test sets as the original dataset
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = 
    train_test_split(X, y, test_size=0.3,
        random_state=21, stratify=y)
```

- Helpful graph for __Exploratory Data Anaylsis (EDA)__
```
_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker='D')
```
![alt text](./images/scatter_matrix.JPG "Scatter Matrix Graph comparing pedal length and width for iris species")


### __K-Nearest Neighbors (KNN)__
- Grab the __k__ closest neighbors and predicting the target value based on those targe values

![alt text](./images/KNN.jpeg "K-Nearest Neighbors Graph with k=3 and k=6")

- Example code snippet
```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(iris['data'],iris['target'])
```
- Overfitting - Underfitting

![alt text](./images/KNN_complexity.jpg "Graph displaying the impact of adjusting k in K-Nearest Neighbors")


