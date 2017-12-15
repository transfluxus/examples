from gensim.models.word2vec import Word2Vec, LineSentence

def word2vec(corpus_file_path, model_file_path):
    model = Word2Vec(size=200,min_count=1, iter=50, window=5, workers=4)
    sentences = LineSentence(corpus_file_path)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.delete_temporary_training_data(True)
    model.save(model_file_path)

path = './data/wikitext-2/merged.txt'

word2vec(path,'wikitext-2.w2v')
