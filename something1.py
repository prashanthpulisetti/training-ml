
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['Star', 'LdaImmutableCurrentContextSlot'],
['ThrowReferenceErrorIfHole' ,'LdaKeyedProperty' ,'LdaNamedProperty'],
['LdaImmutableContextSlot'],
['StackCheck'],
['LdaZero'],
['JumpIfFalse'],
['Constant'],
['Return'],
['CallProperty0'],
['StaKeyedPropertyStrict'],
['CallUndefinedReceiver0'],
['size'],
['Frame'],
['count'],
['Parameter'],
['function'],
['for'],
['bytecode']]
model = Word2Vec(sentences, min_count=1)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()