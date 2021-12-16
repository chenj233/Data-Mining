import pandas as pd

df = pd.read_csv('newtrain.csv')

print(df.columns)
data = df['Text'].tolist()


# lowercase
import string
lower_doc = [doc.lower() for doc in data]
#print(lower_doc)

# tokenization
from nltk.tokenize import word_tokenize
token_doc = [word_tokenize(doc) for doc in lower_doc]
#print(len(token_doc))

# removing punctuation
# cleaning stopwords
stopwords = []
with open('stop_words.lst') as f:
    for line in f:
        stopwords.append(line[:-1])
punctuation = ['(', ')', ':', ';', ',', '.', '/', '"', "'"]
stopwords = stopwords + punctuation


token_doc_no_stopwords = []
# remove stopwords
for doc in token_doc:
    new_term_vec = []
    for word in doc:
        if not word in stopwords:
            new_term_vec.append(word)
    token_doc_no_stopwords.append(new_term_vec)
#print(token_doc_no_stopwords)

# stem mining & Lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
wordnet = WordNetLemmatizer()

preprocessed_docs = []
for doc in token_doc_no_stopwords:
    final_doc = []
    for word in doc:
        #final_doc.append(porter.stem(word))
        final_doc.append(wordnet.lemmatize(word))
    preprocessed_docs.append(final_doc)
#print(preprocessed_docs)

#write to csv

data_dict = {k:v for k,v in enumerate(data)}

for key in data_dict.keys():
    data_dict[key] = preprocessed_docs[key]

s = pd.Series(data_dict)

df['Text'] = s
df.to_csv('newtrain.csv',index = False)

"""
print(len(data))
print(len(preprocessed_docs))

print("--------------------")

print(data[0])
print(preprocessed_docs[0])

import csv

with open('traintest.csv', 'w') as f:
    writer = csv.writer(f, delimiter = ',')

    writer.writerow(['Text'])
    for row in preprocessed_docs:
        writer.writerow([row])

f.close()
"""





