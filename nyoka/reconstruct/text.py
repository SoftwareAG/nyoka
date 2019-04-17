from __future__ import unicode_literals, division

import array
from collections import defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.externals.six.moves import xrange
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.hashing import FeatureHasher
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.utils.validation import check_is_fitted,check_array, FLOAT_DTYPES
from sklearn.utils.fixes import sp_version
#from ..utils.fixes import _Mapping as Mapping  

__all__ = ['CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'TfidfTransformer',
           'TfidfVectorizer',
           'strip_accents_ascii',
           'strip_accents_unicode',
           'strip_tags']


def strip_accents_unicode(s):
    
    normalized = unicodedata.normalize('NFKD', s)
    if normalized == s:
        return s
    else:
        return ''.join([c for c in normalized if not unicodedata.combining(c)])


def strip_accents_ascii(s):
    
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


def strip_tags(s):
    
    return re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", s)


def check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:               # assume it's a collection
        return frozenset(stop)


class VectorizerMixin(object):
    
    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if doc is np.nan:
            raise ValueError("np.nan is an invalid document, expected byte or "
                             "unicode string.")

        return doc

    def _word_ngrams(self, tokens, stop_words=None):
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                ngrams_append(text_document[i: i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in xrange(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        if self.preprocessor is not None:
            return self.preprocessor

        # unfortunately python functools package does not have an efficient
        # `compose` function that would have allowed us to chain a dynamic
        # number of functions. However the cost of a lambda call is a few
        # hundreds of nanoseconds which is negligible when compared to the
        # cost of tokenizing a string of 1000 chars for instance.
        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        if self.lowercase:
            return lambda x: strip_accents(x.lower())
        else:
            return strip_accents

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def get_stop_words(self):
        """Build or fetch the effective stop words list"""
        return check_stop_list(self.stop_words)

    def check_stop_words_consistency(self, stop_words, preprocess, tokenize):
        # NB: stop_words is validated, unlike self.stop_words
        if id(self.stop_words) != getattr(self, '_stop_words_id', None):
            inconsistent = set()
            for w in stop_words or ():
                tokens = list(tokenize(preprocess(w)))
                for token in tokens:
                    if token not in stop_words:
                        inconsistent.add(token)
            self._stop_words_id = id(self.stop_words)

            if inconsistent:
                warnings.warn('Your stop_words may be inconsistent with your '
                              'preprocessing. Tokenizing the stop words '
                              'generated tokens %r not in stop_words.' %
                              sorted(inconsistent))

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            self.check_stop_words_consistency(stop_words, preprocess,
                                               tokenize)
            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in xrange(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't contain index "
                               "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fit-ed)"""
        msg = "%(name)s - Vocabulary wasn't fitted."
        check_is_fitted(self, 'vocabulary_', msg=msg),

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

    def _validate_params(self):
        """Check validity of ngram_range parameter"""
        min_n, max_m = self.ngram_range
        if min_n > max_m:
            raise ValueError(
                "Invalid value for ngram_range=%s "
                "lower boundary larger than the upper boundary."
                % str(self.ngram_range))


class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 non_negative=False, dtype=np.float64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.non_negative = non_negative
        self.dtype = dtype

    def partial_fit(self, X, y=None):
        
        return self

    def fit(self, X, y=None):
        # triggers a parameter validation
        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()

        self._get_hasher().fit(X, y=y)
        return self

    def transform(self, X):
        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()

        analyzer = self.build_analyzer()
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        if self.binary:
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_hasher(self):
        return FeatureHasher(n_features=self.n_features,
                             input_type='string', dtype=self.dtype,
                             alternate_sign=self.alternate_sign,
                             non_negative=self.non_negative)


def document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


class CountVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype

    def sort_features(self, X, vocabulary):
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(six.iteritems(vocabulary))
        map_index = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def limit_features(self, X, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(six.iteritems(vocabulary)):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        if indptr[-1] > 2147483648:  # = 2**31 - 1
            if sp_version >= (0, 14):
                indices_dtype = np.int64
            else:
                raise ValueError(('sparse CSR array has {} non-zero '
                                  'elements and requires 64 bit indexing, '
                                  ' which is unsupported with scipy {}. '
                                  'Please upgrade to scipy >=0.14')
                                 .format(indptr[-1], '.'.join(sp_version)))

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()
        self.validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X = self.sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self.limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)

            self.vocabulary_ = vocabulary

        return X

    def transform(self, raw_documents):
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self.check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X

    def inverse_transform(self, X):
        self.check_vocabulary()

        if sp.issparse(X):
            # We need CSR format for fast row manipulations.
            X = X.tocsr()
        else:
            # We need to convert X to a matrix, so that the indexing
            # returns 2D objects
            X = np.asmatrix(X)
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocabulary = terms[np.argsort(indices)]

        return [inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)]

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if not hasattr(self, 'vocabulary_'):
            self.validate_vocabulary()

        self.check_vocabulary()

        return [t for t, i in sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))]


def make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = document_frequency(X).astype(dtype)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(n_samples / df) + 1
            self.idf_diag = sp.diags(idf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)

        return self

    def transform(self, X, copy=True):
        
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, 'idf_diag', 'idf vector is not fitted')

            expected_n_features = self.idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self.idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self.idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self.idf_diag = sp.spdiags(value, diags=0, m=n_features,
                                    n=n_features, format='csr')


class TfidfVectorizer(CountVectorizer):
    

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(TfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self.tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self.tfidf.norm

    @norm.setter
    def norm(self, value):
        self.tfidf.norm = value

    @property
    def use_idf(self):
        return self.tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self.tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self.tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self.tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self.tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self.tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self.tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self.validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self.tfidf.idf_ = value

    def check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

    def fit(self, raw_documents, y=None):
        
        self.check_params()
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self.tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        
        self.check_params()
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self.tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self.tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        
        check_is_fitted(self, 'tfidf', 'The tfidf vector is not fitted')

        X = super(TfidfVectorizer, self).transform(raw_documents)
        return self.tfidf.transform(X, copy=False)