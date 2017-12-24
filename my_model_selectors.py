import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        leading_score = float('inf')
        best_model = None
        for number_of_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_model = self.base_model(number_of_states)
                log_likelihood = current_model.score(self.X, self.lengths)
                number_of_data_points = sum(self.lengths)
                '''# estimated parameters = Initial state entry probabilities + For each state, for each feature, the mean and variance of the gaussian distribution must be estimated + Transition probabilities between all pairs of states, multiplied by two in both directions (transition-to-self probabilities are known once all other transition probabilities estimated)'''
                number_of_estimated_parameters = number_of_states - 1 + number_of_states*(number_of_states-1) + number_of_states*self.X.shape[1]*2
                bic_score = (number_of_estimated_parameters*np.log(sum(self.lengths))) - (2*log_likelihood)
                if bic_score < leading_score:
                    leading_score = bic_score
                    best_model = current_model
            except:
                pass
        return best_model 


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        leading_score = float('-inf')
        best_model = None
        for number_of_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_model = self.base_model(number_of_states)
                log_likelihood = current_model.score(self.X, self.lengths)
                mean_of_antilikelihoods = np.mean([current_model.score(self.hwords[word][0],self.hwords[word][1]) for word in self.words if word != self.this_word])
                dic_score = log_likelihood - mean_of_antilikelihoods
                if dic_score > leading_score:
                    leading_score = dic_score
                    best_model = current_model
            except:
                pass
        return best_model 


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        leading_score = float('-inf')
        best_model = None
        folder = KFold(n_splits = 3, random_state = None, shuffle = False)
        for number_of_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_model = self.base_model(number_of_states)
                sum_log_likelihood_folds = 0
                if len(self.sequences) < 3:
                    sum_log_likelihood_folds = current_model.score(self.X, self.lengths)
                else:
                    for train_seq, test_seq in folder.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train_seq, self.sequences)
                        test_data, test_lengths = combine_sequences(test_seq, self.sequences)
                        folded_current_model = self.base_model(number_of_states)
                        sum_log_likelihood_folds += folded_current_model.score(test_data, test_lengths)
                if sum_log_likelihood_folds > leading_score:
                    leading_score = sum_log_likelihood_folds
                    best_model = current_model
                self.X, self.lengths = self.hwords[self.this_word]
                sum_log_likelihood_folds = 0
            except:
                pass
        return best_model 
