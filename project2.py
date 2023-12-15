import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy

'''
 References:
 1. Word embedding: https://www.youtube.com/watch?v=viZrOnJclY0
 2. word2vec:https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
 3. spectral clustering: https://zhuanlan.zhihu.com/p/392736238
 TODO:
 1. How to prove the presence of bias
 2. Trained model only uses 1 CSV file. We should try the performance on 13 files.
 3. Filter training dataset based on our theme.
'''

# Load your CSV file
# Replace 'your_dataset.csv' with the actual path or URL to your CSV file
csv_file_path = './russian-troll-tweets/IRAhandle_tweets_1.csv'
df = pd.read_csv(csv_file_path)

# Check the structure of your CSV file
print(df.head())

# Assuming your CSV file has a column named 'text' containing the text data
text_data = df['content'].tolist()
# Download NLTK resources
'''
Stopwords are common words that are often filtered out during 
text processing because they are considered to be of little 
value in terms of conveying meaningful information. 
Examples of stopwords include words like "the," "and," "is," "in," etc.
'''
nltk.download('stopwords')
'''
Break the texts into sentences
'''
nltk.download('punkt')

# Load your text data
# text_data = ...

# Preprocess the text
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return words


def run_word2vec_model(data, sub_words, n_clusters):
    tokenized_data = [preprocess_text(text) for text in data]
    # Train Word2Vec embeddings
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    # IMPORTANT: Adjust the subset size as needed
    subset_embeddings = model.wv.vectors[:100]
    draw(n_clusters, subset_embeddings)


def run_pretrained_model(sub_words, n_clusters):
    # Load pre-trained embeddings (example using GloVe)
    # Load GloVe pre-trained embeddings
    tokenized_data = [preprocess_text(text) for text in text_data]

    # Load pre-trained embeddings (GloVe in this example)
    embedding_path = './glove.6B.100d.txt'
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Map words to their pre-trained vectors
    pretrained_embeddings = [embeddings_index[word] for words in tokenized_data for word in words if
                             (word in embeddings_index and word in sub_words)]
    # Convert the list to a numpy array
    pretrained_embeddings = numpy.array(pretrained_embeddings)
    print(pretrained_embeddings)
    draw(n_clusters, pretrained_embeddings)


def draw(n_clusters, subset_embeddings):
    # Perform Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors")
    cluster_labels = spectral.fit_predict(subset_embeddings)
    # Visualize Clustering using t-SNE
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(subset_embeddings)

    # Plot the clusters
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('t-SNE Visualization of Clusters')
    plt.show()


# Choose a subset of words for analysis
# Replace with the words you want to include in the subset
subset_words = [
    "fireman",
    "stewardess",
    "waitress",
    "businessman",
    "mailman",
    "chairman",
    "policeman",
    "salesman",
    "mankind",
    "forefather",
    "gentleman",
    "master",
    "mistress",
    "careerwoman",
    "tomboy",
    "girly",
    "he",
    "she",
    "man-made",
    "manpower",
    "girl Friday",
    "gal",
    "dude",
    "effeminate",
    "virile",
    "frail",
    "hysterical",
    "bitchy",
    "bossy",
    "macho",
    "queen bee",
    "catfight",
    "damsel",
    "gentleman caller",
]

run_word2vec_model(text_data, subset_words, 3)
# run_pretrained_model(subset_words, 3)
