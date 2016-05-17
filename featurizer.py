#!/usr/bin/env python3

import numpy as np
from scipy.sparse import linalg as spla, hstack, issparse, csr_matrix
from math import ceil
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk      
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
		return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]

class NETokenizer():
	""" Named Entity tokenizer. Removes the named
	entities from sentences and replaces them with their labels.
	"""
	def __init__(self):
		pass
	def __call__(self, doc):
		""" Tokenizes the input.
		If a word is not a named entity, tokenizes it as it is.
		Else, tokenizes the word as <label>_<tag> (ex: PERSON_NNP)
		"""
		tokenized = nltk.word_tokenize(doc)
		tagged = nltk.pos_tag(tokenized)
		namedEnt = nltk.ne_chunk(tagged)
		return [(e.label()+'_'+e[0][1]) if isinstance(e,nltk.tree.Tree) else e[0] for e in namedEnt]

def jaccard(A,B):
	""" Returns the Jaccard coefficient between two sets. """
	return len(A & B) / len(A | B)

def cosine_simil(A, B):
	""" Returns the cosine similarity between the rows of A
	and B. The output is a Mx1 vector where M is the number
	of rows of A. A and B are sparse matrices.
	"""
	if len(A.shape) == 1:
		A = A.reshape((1,A.shape[0]))
	if len(B.shape) == 1:
		B = B.reshape((1,B.shape[0]))
	C = A.dot(B.T)
	# Reshape into a column vector
	AN = spla.norm(A, axis = 1).reshape((A.shape[0],1)) if issparse(A) else np.linalg.norm(A, axis = 1).reshape((A.shape[0],1))
	BN = spla.norm(B) if issparse(B) else np.linalg.norm(B)
	if issparse(C):
		C = C.multiply(1. / (AN * BN))
	else:
		if len(C.shape) == 1:
			C = C.reshape(C.shape[0],1)
		AN *= BN
		C /= AN
	return C

def cosine_simil2(A, B):
	""" Returns the cosine similarity between the rows of A
	and B. The output is a Mx1 vector where M is the number
	of rows of A. A and B are sparse matrices.
	"""
	C = A.dot(B.T)
	AN = spla.norm(A, axis = 1).reshape((A.shape[0],1)) # Reshape into a column vector
	BN = np.linalg.norm(B)
	C = C * (1. / (AN * BN))
	return C

class Featurizer():
	def __init__(self, plot_vectorizer = 'count', tokenizer = None, lda = False, use_genre_vecs = False):
		t = None
		if tokenizer is 'named_entity':
			t = NETokenizer()
		elif tokenizer is 'lemma':
			t = LemmaTokenizer()
		self.use_genre_vecs = use_genre_vecs
		self.binary = plot_vectorizer is 'binary'
		if plot_vectorizer is 'tfidf':
			self.vectorizer = TfidfVectorizer(analyzer = "word",   \
				tokenizer = t,    \
				preprocessor = None, \
				stop_words = 'english')
		elif plot_vectorizer is 'binary':
			self.vectorizer = CountVectorizer(analyzer = "word",	\
				tokenizer = t,	\
				preprocessor = None, \
				stop_words = 'english', \
				binary = True)
		else:
			self.vectorizer = CountVectorizer(analyzer = "word",   \
				tokenizer = t,    \
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

	def load(self, path):
		""" Loads the data into memory. """
		with io.open(path, 'r', encoding = 'latin-1') as f:
			movies = json.load(f)
			od = OrderedDict({(movie['title'],movie['year']):{'plot':movie['plot'],'cast':set(movie['cast']), \
				'genres':set(movie['genres'])} \
				for movie in movies}.items())
			return od

	def train(self, movies):
		""" Trains the featurizer. """
		movie_keys = list(movies.keys())
		self.movies = dict(zip(movie_keys, range(0, len(movie_keys))))
		self.movie_indices = dict([reversed(i) for i in self.movies.items()])
		plots = [movie['plot'] for movie in movies.values()]
		self.plots = self.vectorizer.fit_transform(plots)
		self.casts = [movie['cast'] for movie in movies.values()]
		self.genres = [movie['genres'] for movie in movies.values()]
		if self.lda is not None:
			self.plot_topics = self.lda.fit_transform(feat_vec)
		else:
			self.plot_topics = None

		if self.use_genre_vecs:
			genre_lis = set([])
			for g in self.genres:
				genre_lis.update(g)
			self.genre_lis = dict(zip(genre_lis, range(0, len(genre_lis))))
			self.genre_indices = dict([reversed(i) for i in self.genre_lis.items()])
			genre_plots = np.zeros((len(genre_lis),self.plots.shape[1]))
			for i in range(len(self.genres)):
				gl = self.genres[i]
				for g in gl:
					genre_plots[self.genre_lis[g],:] += self.plots[i,:]
			if self.binary:
				genre_plots = np.minimum(np.ones((len(genre_lis),self.plots.shape[1])),genre_plots)
			self.genre_plots = cosine_simil(self.plots, genre_plots)

	def load_train(self, path):
		""" Loads the data into memory and trains the featurizer. """
		self.train(self.load(path))


	def plot_features(self, base_movie, plots, plot_topics = None):
		""" Returns a feature matrix derived from the plots.
		The # of rows returned matches the length of the parameter plots.
		"""
		if self.use_genre_vecs:
			plot = self.genre_plots[self.movies[base_movie]]
			pv = cosine_simil(plots, plot)
			return pv
		else:
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
		return self.features(base_movie, movies = ((self.genre_plots[ind] if self.use_genre_vecs else self.plots[ind], self.plot_topics[ind] if self.lda is not None else None), [self.casts[ind]], [self.genres[ind]]))

	def features(self, base_movie, movies = None):
		""" Returns the feature set for the given movies,
		when compared to the base movie. When movies is None,
		uses the whole list of movies.

		Parameter movies must be a 3-tuple, representing the plots,
		casts and genres. The # of rows of each should match.

		Returns an AxB matrix where A is the # of rows for plots
		and B is the total number of features.
		"""
		plots = (self.genre_plots if self.use_genre_vecs else self.plots) if movies is None else movies[0][0]
		plot_topics = self.plot_topics if movies is None else movies[0][1]
		casts = self.casts if movies is None else movies[1]
		genres = self.genres if movies is None else movies[2]
		pv = self.plot_features(base_movie, plots, plot_topics)
		cv = self.cast_features(base_movie, casts)
		gv = self.genre_features(base_movie, genres)
		return hstack((pv,cv,gv)) if issparse(pv) else np.hstack((pv,cv,gv))

	def similar_movies(self, weights, base_movie, movies = None, n = 6):
		""" Gets the n similar movies to a base movie. """
		fv = self.features(base_movie, movies = movies)
		wv = weights.reshape((weights.shape[1],1))
		scores = fv.dot(wv)
		inds = np.argpartition(scores,-n, axis = 0)[-n:].reshape(n)
		return [self.movie_indices[i]for i in inds]

if __name__ == '__main__':
	cf = Featurizer(plot_vectorizer = 'count', tokenizer = None, lda = False, use_genre_vecs = True)
	cf.load_train('data.json')
	q = cf.find_movie('Fast and')
	#f = cf.single_features(q[0],q[1])
	f = cf.features(q[0])
	print(f.shape)
	sm = cf.similar_movies(np.array([-0.96477944, 30.29397824, -0.64196636]).reshape((1,3)), q[0])
	print(sm)