import numpy as np
from data import data_train, data_test
# BoW features
from sklearn.feature_extraction.text import CountVectorizer


class Perceptron(object):
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs

    def predict(self, X):
        preds = np.dot(np.expand_dims(self.w_, axis=0), X.T)
        return np.where(preds >= 0, 1, 0)
    # Rosenblatt
    def train(self, X, y):
        # Initialize weights
        self.w_ = np.random.randn(X.shape[1])
        self.errors = []
        for epoch in range(self.epochs):
            err_count = 0
            # Calculate classification errors
            for xi, yi in zip(X, y):
                xi = np.squeeze(xi.toarray())
                output = np.dot(self.w_, xi) > 0
                if output == yi:
                    continue
                err_count += 1
                # calculate update
                update = self.lr * (yi - output) * xi
                # Update w
                self.w_ += update
            print("Epoch {0} Errors {1}".format(epoch, err_count))
            self.errors.append(err_count)
        return self


bow_converter = CountVectorizer(lowercase=True, max_features=1000)
doc_vecs = bow_converter.fit_transform(data_train.data)

# feature index
print(bow_converter.vocabulary_.get(u'baseball'))

# doc_vecs.shape (documents, features)

perceptron = Perceptron(lr=0.1, epochs=60)
perceptron.train(X=doc_vecs, y=data_train.target)

test_doc_vecs = bow_converter.transform(data_test.data)
preds = perceptron.predict(X=test_doc_vecs.toarray())
print("Incorrect Predictions (Test Data) {0}%".format(
    np.round(np.sum(preds != data_test.target) / len(data_test.data), 2) * 100))
