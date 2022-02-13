import numpy as np

class LogisticRegressor:
    def __init__(self, lr=0.1, epochs=10):
        # epochs --> Number of iterations.
        self.epochs = epochs
        # lr --> Learning rate.
        self.lr = lr

    # Binary Cross-Entropy Loss Function
    def loss(self, y, y_hat):
        epsilon = 1e-5
        loss = -np.mean(y * (np.log(y_hat + epsilon)) - (1 - y) * np.log(1 - y_hat + epsilon))
        return loss


    def sigmoid(self,z):
        return 1.0 / (1 + np.exp(-z))

    def gradient(self, X, y, y_hat):
        n = X.shape[0]
        d = (y_hat - y)
        # Gradient of loss w.r.t weights.
        self.dw = (1 / n) * np.dot(X.T, d)
        # Gradient of loss w.r.t bias.
        self.db = (1 / n) * np.sum(d)

    def predict(self, X):
        wTx = np.dot(X.toarray(), self.w) + self.b
        y_hat = self.sigmoid(wTx) > 0.5
        return y_hat


    def train(self, X, y):
        # X --> Input.
        # y --> true/target value.

        # n-> number of training examples
        # m-> number of features
        n, m = X.shape

        # Initializing weights and bias to zeros.
        self.w = np.zeros((m, 1))
        self.b = 0.0

        # Reshaping y.
        y = y.reshape(n, 1)

        # Errors/losses.
        losses = []

        # Training loop.
        for epoch in range(self.epochs):
            bs = 64
            for i in range((n - 1) // bs + 1):
                # Defining batches. SGD.
                start_i = i * bs
                end_i = start_i + bs
                xb = X[start_i:end_i].toarray()
                yb = y[start_i:end_i]

                # linear transformation
                wTx = np.dot(xb, self.w) + self.b
                # non-linearity
                y_hat = self.sigmoid(wTx)

                # Getting the gradients of loss w.r.t parameters.
                self.gradient(xb, yb, y_hat)

                # Updating the parameters.
                self.w -= self.lr * self.dw
                self.b -= self.lr * self.db
            # Calculating loss and appending it in the list.
            wTx = np.dot(X.toarray(), self.w) + self.b
            y_hat = self.sigmoid(wTx)
            l = self.loss(y, y_hat)
            print("epoch {0}, loss {1}".format(epoch, l))
            losses.append(l)

        # returning weights, bias and losses(List).
        return self.w, self.b, losses


from data import data_train, data_test
# BoW features
from sklearn.feature_extraction.text import CountVectorizer

bow_converter = CountVectorizer(lowercase=True, max_features=500)
doc_vecs = bow_converter.fit_transform(data_train.data)

# feature index
print(bow_converter.vocabulary_.get(u'baseball'))

# doc_vecs.shape (documents, features)

lr = LogisticRegressor(lr=0.1, epochs=60)
lr.train(X=doc_vecs, y=data_train.target)

test_doc_vecs = bow_converter.transform(data_test.data)
preds = lr.predict(X=test_doc_vecs)
print("Incorrect Predictions (Test Data) {0}%".format(
    np.round(np.sum(preds != data_test.target) / len(data_test.data), 2) * 100))