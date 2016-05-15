#!/usr/bin/env python3

import numpy as np
from scipy.sparse import linalg as spla, hstack
from math import ceil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
import io, json
from collections import OrderedDict


class LemmaTokenizer(object):
	""" Define the Lemma Tokenizer class to use
	if the user wishes to lemmatize the plot.
	"""
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def jaccard(A,B):
	""" Returns the Jaccard coefficient between two sets. """
	return len(A & B) / len(A | B)

def cosine_simil(A, B):
	""" Returns the cosine similarity between the rows of A
	and B. The output is a Mx1 vector where M is the number
	of rows of A. A and B are sparse matrices.
	"""
	C = A.dot(B.T)
	AN = spla.norm(A, axis = 1).reshape((A.shape[0],1)) # Reshape into a column vector
	BN = spla.norm(B)
	C = C.multiply(1. / (AN * BN))
	return C

class Featurizer():
	def __init__(self, plot_vectorizer = 'count', lemmatize = False, lda = False):
		if plot_vectorizer is 'tfidf':
			self.vectorizer = TfidfVectorizer(analyzer = "word",   \
				tokenizer = LemmaTokenizer() if lemmatize else None,    \
				preprocessor = None, \
				stop_words = 'english')
		else:
			self.vectorizer = CountVectorizer(analyzer = "word",   \
				tokenizer = LemmaTokenizer() if lemmatize else None,    \
				preprocessor = None, \
				stop_words = 'english')
		if lda:
			self.lda = LatentDirichletAllocation(n_topics=20, max_iter=2,	\
				learning_method='online', learning_offset=10.,	\
				random_state=0)
		else:
			self.lda = None

	def find_movie(self, title, year = None):
		""" Finds a movie with the given name substring. """
		return [movie for movie in self.movies.keys() if title in movie[0] and (year is None or year == movie[1])]

	def load_train(self, path):
		""" Loads the data into memory and trains the featurizer. """
		with io.open(path, 'r', encoding = 'latin-1') as f:
			movies = json.load(f)
			od = OrderedDict({(movie['title'],movie['year']):{'plot':movie['plot'],'cast':set(movie['cast']), \
				'genres':set(movie['genres'])} \
				for movie in movies}.items())
			self.train(od)

	def train(self, movies):
		movie_keys = list(movies.keys())
		self.movies = dict(zip(movie_keys, range(0, len(movie_keys))))
		self.movie_indices = dict([reversed(i) for i in self.movies.items()])
		plots = [movie['plot'] for movie in movies.values()]
		self.plots = self.vectorizer.fit_transform(plots)
		self.casts = [movie['cast'] for movie in movies.values()]
		self.genres = [movie['genres'] for movie in movies.values()]
		if self.lda is not None:
			self.plot_topics = self.lda.fit_transform(feat_vec)

	def plot_features(self, base_movie, plots):
		""" Returns a feature matrix derived from the plots.
		The # of rows returned matches the length of the parameter plots.
		"""
		plot = self.plots[self.movies[base_movie]]
		pv = cosine_simil(plots, plot)
		return pv

	def cast_features(self, base_movie, casts):
		""" Returns a feature matrix derived from the casts.
		The # of rows returned matches the length of the parameter casts.
		"""
		cv = np.array([jaccard(cast_set, self.casts[self.movies[base_movie]]) for cast_set in casts])
		return cv.reshape((cv.shape[0],1)) # Reshape into column vector

	def genre_features(self, base_movie, genres):
		""" Returns a feature matrix derived from the genres.
		The # of rows returned matches the length of the parameter genres.
		"""
		gv = np.array([jaccard(genre_set, self.genres[self.movies[base_movie]]) for genre_set in genres])
		return gv.reshape((gv.shape[0],1)) # Reshape into column vector

	def single_features(self, base_movie, trial_movie):
		""" Returns a feature matrix for a single movie. """
		ind = self.movies[trial_movie]
		return self.features(base_movie, movies = (self.plots[ind], [self.casts[ind]], [self.genres[ind]]))

	def features(self, base_movie, movies = None):
		""" Returns the feature set for the given movies,
		when compared to the base movie. When movies is None,
		uses the whole list of movies.

		Parameter movies must be a 3-tuple, representing the plots,
		casts and genres. The # of rows of each should match.

		Returns an AxB matrix where A is the # of rows for plots
		and B is the total number of features.
		"""
		plots = self.plots if movies is None else movies[0]
		casts = self.casts if movies is None else movies[1]
		genres = self.genres if movies is None else movies[2]
		pv = self.plot_features(base_movie, plots)
		cv = self.cast_features(base_movie, casts)
		gv = self.genre_features(base_movie, genres)
		return hstack((pv,cv,gv))

if __name__ == '__main__':
	cf = Featurizer(plot_vectorizer = 'count', lemmatize = False, lda = False)
	cf.load_train('data.json')
	q = cf.find_movie('Fast and')
	#f = cf.single_features(q[0],q[1])
	f = cf.features(q[0])
	print(f)
	print(f.shape)