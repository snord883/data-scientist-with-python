## __CASE STUDY: SCHOOL BUDGETING__
[Full Jupyter Notebook](https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/notebooks/1.0-full-model.ipynb)

### __Notes:__
- Encode labels as categories
```
categorize_label = lambda x: x.astype('category')
df.label = df[['label']].apply(categorize_label, axis=0)
```
- **Log loss**
    - y=1 => Correct
    - y=0 => Wrong
    - p => percentage of confidence
    - Loss is larger for wrong prediction with high confidence
    - Loss is smaller for correct prediction with high confidence
```
logloss = y*log(p) + (1-y)*log(1-p)
```

- Preprocessing multiple dtypes
    - Problem: Pipeline steps for numeric and text preprocessing can't follow each other
        - e.g., output of *CountVectorizer* can't be input for *Imputer*
    - Solution: 
        - **FunctionTransformer()**
            - Turns Python function into an object that a scikit-learn Pipeline can understand
            ```
            from sklearn.preprocessing import FunctionTransformer
            get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
            get_numeric_data = FunctionTransformer(lambda x: x[['numeric','with_missing']], validate=False)
            ```
            - *validate=False* tells scikit-learn it doesn't need to check for NaNs or validate the dtypes of the input
        - **FeatureUnion()**
            - Takes the two separate outputs from *FunctionTransformer* and puts them together as a single array for the input to the classifier
            ```
            from sklearn.preprocessing import FunctionUnion
            union = FeatureUnion([
                ('numeric', numberic_pipeline),
                ('text', text_pipeline)
            ])
            ```
    - Putting it all together:
        ```
        numeric_pipeline = Pipeline([
                                ('selector', get_numeric_data),
                                ('imputer', Imputer())
                            ])

        text_pipeline = Pipeline([
                                ('selector', get_text_data),
                                ('vectorizer', CountVectorizer())
                            ])

        pl = Pipeline([
                ('union', union),
                ('clf', OneVsRestClassifier(LogRegression))
            ])
        ```

- Expert Tips:
    - **interaction terms**:
        ```
        from sklearn.preprocessing import PolynomialFeatures
        interaction = PolynomialFeatures(
                            degree=2,
                            interaction_only=True,
                            include_bias=False
        )
        ```
        - *interaction_only=True* means scikit-learn will NOT include interactions with itself (some_column x some_column) 
        - *CountVectorizer* returns a sparce matrix, and the standard PolynomialFeatures is NOT compatible with a sparce matrix
            - Alternative: [here](https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/features/SparseInteractions.py) 

    ![alt text](.\images\interaction_terms.JPG "image")

    - **hashing**:
        ```
        from sklearn.feature_extraction.text import HashingVectorizer
        vec = HashingVectorizer(norm=None,
                                non_negative=True,
                                token_pattern=TOKEN_ALPHANUMERIC,
                                ngram_range=(1,2))
        ```