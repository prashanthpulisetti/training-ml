from gensim.models import Word2Vec
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
# train model
model = Word2Vec(sentences, min_count=1)
print(model)
words = list(model.wv.vocab)
print(words)
print(model['bytecode'])
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)