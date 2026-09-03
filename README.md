# Fake News Prediction

A compact natural-language-processing (NLP) baseline for classifying news articles
as **FAKE** or **REAL**. The project is implemented as a Jupyter notebook and uses
TF-IDF features with logistic regression. It is intended as an educational example
of a complete text-classification workflow: loading data, normalizing text,
vectorizing documents, training a model, evaluating it, and predicting an unseen
test example.

> **Important:** this is a binary classifier trained on a specific labelled dataset;
> it is **not** a fact-checking system. A predicted label should never be used as
> evidence that a claim is true or false.

## Repository contents

| Path | Purpose |
| --- | --- |
| `fake_news_prediction.ipynb` | The complete, executable implementation and walkthrough. |
| `sample_data/fake_or_real_news.csv` | Labelled article data consumed by the notebook. |
| `README.md` | Project overview, setup, and implementation notes. |

## Dataset

The supplied CSV contains 6,335 labelled articles. Its columns are:

| Column | Description | Used by the model? |
| --- | --- | --- |
| unnamed first column | Source row/index identifier. | No. |
| `title` | Article headline. | Yes. |
| `text` | Article body. | Yes. |
| `label` | Ground-truth class: `FAKE` or `REAL`. | Yes, as the target. |

The included data is nearly balanced: 3,164 `FAKE` rows and 3,171 `REAL` rows.
There are no empty values in these four CSV fields. The notebook combines `title`
and `text` rather than modelling them separately.

## How the implementation works

The notebook implements the following pipeline.

```text
CSV article data
    │
    ├── join title + text into `content`
    ├── remove non-letters, lowercase, tokenize
    ├── remove English stop words
    ├── Porter stem each remaining token
    │
    ├── TF-IDF vectorization
    ├── stratified 80% training / 20% test split (random_state=2)
    ├── logistic-regression training
    │
    └── accuracy evaluation and single-example prediction
```

### 1. Text construction and labels

For every row, the notebook creates `content` by concatenating the headline and
body. Labels are converted from strings to integers:

* `FAKE` → `0`
* `REAL` → `1`

The extra CSV index column remains in the loaded DataFrame but is not included in
the final text feature matrix.

### 2. Normalization and stemming

The `stemming` function applies these operations in order:

1. Replaces every non-alphabetic character with a space.
2. Converts the text to lowercase and splits it into tokens.
3. Removes NLTK's English stop words (for example, common function words such as
   “the” and “and”).
4. Applies NLTK's `PorterStemmer` to reduce related word forms to a shared stem.
5. Joins the resulting tokens back into one normalized document.

For example, terms such as `running`, `runs`, and `runner` may be reduced to a
common stem. This reduces vocabulary size, although it can also make text less
readable and discard useful linguistic distinctions.

### 3. TF-IDF features

`sklearn.feature_extraction.text.TfidfVectorizer` converts each processed document
into a sparse numeric vector. A feature receives more weight when it is important
in one document but less common across the collection. This representation is a
strong, interpretable baseline for document classification and avoids constructing
a dense matrix for a large vocabulary.

### 4. Training and evaluation

The data is split with `train_test_split(test_size=0.2, stratify=Y,
random_state=2)`, preserving the class ratio in both partitions. A scikit-learn
`LogisticRegression` classifier is then fitted on the training vectors. The
notebook reports accuracy for both training and held-out test data.

The saved notebook outputs report:

| Split | Accuracy |
| --- | ---: |
| Training | 95.28% |
| Test | 91.55% |

These figures are a record of the notebook's existing run, not a guarantee for
future runs, different library versions, or new data.

## Requirements

* Python 3.9 or newer (a recent Python 3 release is recommended).
* Jupyter Notebook or JupyterLab.
* `numpy`
* `pandas`
* `nltk`
* `scikit-learn`

## Run the project

From the repository root, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas nltk scikit-learn
```

Start Jupyter from the repository root so the notebook's relative data path,
`./sample_data/fake_or_real_news.csv`, resolves correctly:

```bash
jupyter notebook
```

Open `fake_news_prediction.ipynb` and run its cells from top to bottom. On first
use, the notebook downloads the NLTK `stopwords` corpus via
`nltk.download('stopwords')`; this requires network access unless the corpus is
already installed locally.

### Run in a non-interactive environment

After installing the packages and NLTK corpus, execute the whole notebook with:

```bash
jupyter nbconvert --to notebook --execute --inplace fake_news_prediction.ipynb
```

This updates the notebook's cell outputs in place. Omit `--inplace` and specify an
output file if you want to preserve the checked-in outputs.

## Make a prediction

The final notebook cells demonstrate a prediction using one row from `X_test`:

```python
X_new = X_test[10]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("This news is fake")
else:
    print("This news is true")
```

To classify new article text, apply the *same* preprocessing function and the
already fitted vectorizer before calling the model. Do not fit a new vectorizer for
each prediction, because its vocabulary and feature positions would no longer
match the trained model:

```python
article = "Headline and article body go here."
article_features = vectorizer.transform([stemming(article)])
predicted_label = model.predict(article_features)[0]
print("REAL" if predicted_label == 1 else "FAKE")
```

## Limitations and recommended next steps

This notebook is intentionally a simple baseline. Before using it beyond learning
or demonstration, consider the following improvements:

1. **Prevent evaluation leakage.** The current notebook fits `TfidfVectorizer` on
   all documents before the train/test split. Fit it on `X_train` only, then use
   `.transform()` for `X_test`, or use a scikit-learn `Pipeline`.
2. **Use richer evaluation.** Report precision, recall, F1 score, a confusion
   matrix, and cross-validation results in addition to accuracy.
3. **Tune the model.** Evaluate logistic-regression regularization and class
   weights with a validation procedure rather than relying on defaults.
4. **Validate real-world generalization.** Random article splits can overstate
   quality when related sources or near-duplicate stories appear in both sets.
   Consider time-based, source-based, or deduplicated splits.
5. **Handle provenance and bias.** The model learns correlations in the supplied
   labels and writing style; it does not verify sources, citations, or claims.
   Review dataset licensing, label quality, representativeness, and potential
   demographic or political bias before any deployment.
6. **Persist the trained pipeline.** Save a jointly fitted preprocessing/vectorizer
   and classifier artifact (for example, with `joblib`) along with dependency and
   dataset versions for reproducible inference.

## Reproducibility notes

The train/test split is deterministic because the notebook sets `random_state=2`.
Results can nevertheless change after modifying the dataset, preprocessing,
scikit-learn or NLTK versions, or model hyperparameters. For a reproducible
experiment, record the Python package versions, dataset version, and exact
notebook execution environment alongside the reported metrics.

## Responsible use

Treat outputs as model estimates rather than factual determinations. Keep a human
reviewer in the loop, verify claims through reputable primary sources and
professional fact-checking organizations, and do not use this baseline as the sole
basis for moderation, publishing, legal, financial, medical, employment, or other
high-impact decisions.
