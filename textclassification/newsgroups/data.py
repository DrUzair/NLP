from sklearn.datasets import fetch_20newsgroups

# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
# 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
# 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
# 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

categories = ['talk.politics.misc', 'rec.sport.baseball']
print("Loading 20 newsgroups dataset for categories:", categories)

data_train = fetch_20newsgroups(
    subset="train", categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes")
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove= ("headers", "footers", "quotes")
)

