from gensim.corpora import WikiCorpus, Dictionary
from gensim.utils import to_unicode, simple_preprocess

def tokenize(text):
    return simple_preprocess(text)
in_f = "/notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2"
out_f = "/notebooks/embedding/data/processed/processed_wiki_ko.txt"
output = open(out_f, 'w')
wiki = WikiCorpus(in_f, tokenizer_func=tokenize, dictionary=Dictionary())
i = 0
for text in wiki.get_texts():
    output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
    i = i + 1
    if (i % 10000 == 0):
        print('preocessed' + str(i) + 'articles')
output.close()       

#/notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2

