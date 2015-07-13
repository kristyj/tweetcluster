import math


################# OVERALL LIKELIHOOD EQNS #######################
# eg my_lm.lm_eqn

def mle_eqn(my_lm, order, fullterm):
    """Basic Maximum-Likelihood Language Model calculation.
    len(fullterm) must equal order given.
    Returns probability of fullterm[-1] given fullterm[:-1] in linear space."""
    #print('fullterm', fullterm)
    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)
    #print('hist',history)
    return my_lm.ngrams[order].get(fullterm, 0) \
           / \
           my_lm.ngrams[order - 1].get(history,0)

def add_one_eqn(my_lm, order, fullterm):
    """Basic Add-one language model calculation, assuming that every unknown term occurs once,
    and adding one to each unigram count.
    Returns probability of fullterm[-1] given fullterm[:-1] in linear space. """
    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)
    # assume unigram count of 1 for every term
    unigram_vocab = my_lm.vocabsize[1]
    # raise to power of order for potential vocab at chosen order
    potential_ngrams = math.pow(unigram_vocab, order)
    return  (my_lm.ngrams[order].get(fullterm,0) + 1) \
            / \
            (my_lm.ngrams[order - 1].get(history,0) + potential_ngrams)

def add_alpha_eqn(my_lm, order, fullterm, alpha=0.05):
    """Basic add-alpha LM calculation, assumes each unknown term occurs alpha times, adds alpha to each unigram.
    Returns probability of fullterm[-1] given fullterm[:-1] in linear space."""
    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)
    unigram_vocab = my_lm.vocabsize[1]
    potential_ngrams = math.pow(unigram_vocab, my_lm.order)
    return  (my_lm.ngrams[order].get(fullterm,0) + alpha) \
            / \
            (my_lm.ngrams[order - 1].get(history,0) + (alpha * potential_ngrams))

def good_turing_eqn(my_lm, order, fullterm):
    """Good Turing LM calculation - like MLE but with expected count rather than actual.
    eturns probability of fullterm[-1] given fullterm[:-1] in linear space."""
    # TODO: if param_eqn should return prob for OOV
    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)

    unigram_vocab = my_lm.vocabsize[1]
    potential_ngrams = math.pow(unigram_vocab, my_lm.order)

    original_count = my_lm.ngrams[order].get(fullterm, 0)
    expected_count = (original_count +1) * \
                     (my_lm.count_of_counts[order].get(original_count + 1, 0)
                      /
                      (my_lm.count_of_counts[order].get(original_count, potential_ngrams) ))
    return expected_count \
           / \
           my_lm.ngrams[order - 1][history]


############# INTERPOLATION AND BACKOFF, RELYING ON MLE/PROB FNS ABOVE #################
# eg my_lm.smoothing_eqn

def linear_interpolation(my_lm, order, fullterm):
    """Linear interpolation calculation for the term, relying on the interpolation parameters first being set.
    Parameters are in mylm.params.parameters.
    Relies on another equation being available for the probability of the ngrams|history, eg MLE.
    Probs and params are two vectors in ascending order from 1 to order, dot product calculated.
    Returns probability of fullterm[-1] given fullterm[:-1] in linear space."""

    # TODO: Parameters should be set optimized on held-out set, not yet performed here.
    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)

    #
    # get the probabilitity of the word | history at order specified
    probs_by_order = [my_lm.lm_params.lm_eqn(my_lm, my_lm.order, history[order-x:], word) for x in range(1, order+1)]
    # get the interpolation parameters
    params = my_lm.lm_params.parameters

    # check parameters are correct, perform calculation
    if len(params) != len(probs_by_order):
        print("There are more lambda values than probabilities from counts!")
    elif abs(sum(params)-1) > 0.001:
        print("Smoothing params sum to a number other than one!")
    else:
        return sum([params[x] *  probs_by_order[x] for x in range(1, order+1)])


def recursive_interpolation(my_lm, order, fullterm):
    """Recursive interpolation LM calculation.
    Makes use of pre-computed parameters about how much each backoff stage is trusted.
    Returns probability of fullterm[-1] given fullterm[:-1] in linear space."""

    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)

    # get probabilities and parameters as vectors in ascending order from 1 to n
    probs_by_order = [my_lm.lm_params.lm_eqn(my_lm, my_lm.order, history[order-x:], word) for x in range(1, order+1)]
    params = my_lm.lm_params.parameter

    # check values and get probability as recursive function
    if len(params) != len(probs_by_order):
        print("There are more lambda values than probabilities from counts!")
    elif abs(sum(params)-1) > 0.001:
        print("Smoothing params sum to a number other than one!")
    else:
        def recur_int(probs_by_order, params):
            if len(probs_by_order) == 1: # only unigram probability left
                return probs_by_order.pop() * params.pop()
            else:
                lam_val = params.pop()
                return lam_val* probs_by_order.pop() + (1 - lam_val)* recur_int(probs_by_order, params)

        return recur_int(probs_by_order, params)

def recursive_backoff(my_lm, order, fullterm, d):
    """Recursive backoff LM calculation.
    Raw LM probabilities are discounted by 0 < d < 1, and remaining probability is given to BO model.
    D should be set in some principled way, eg using Good-Turing discounting.
    Returns probability of fullterm[-1] given fullterm[:-1] in linear space."""
    # TODO: This is only a temporary equation - in reality d should be conditioned on the history, not just set.

    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)

    counts_by_order = [my_lm.ngrams[x].get((history[order-x:]+ word)) for x in range(1, order+1)]
    probs_by_order = [my_lm.lm_params.lm_eqn(my_lm, my_lm.order, history[order-x:], word) for x in range(1, order+1)]


    ds = my_lm.lm_params.param_eqn(my_lm, order, fullterm) # eg uses good-turing smoothing and zero-mass is given to backoff
                                                            # or eg uses witten-bell smoothing

    def recur_bo(counts_by_order, probs_by_order, d_list):
        current_count = counts_by_order.pop()
        current_prob = probs_by_order.pop()
        d = d_list.pop()

        if current_count > 0:
            return d * current_prob

        else: # TODO: Check that alpha is 1-d
            return (1-d) * recur_bo(counts_by_order, probs_by_order, d)
    return recur_bo(counts_by_order, probs_by_order, ds)



################# SMOOTHING PARAMETER ESTIMATION EQNS ######################
# eg my_lm.param_eqn

def witten_bell(my_lm, order, fullterm):
    history, word = tuple(fullterm[:-1]), tuple(fullterm[-1])
    fullterm = tuple(fullterm)

    # parameters as ascending list from 1 to order
    def num_extensions(my_lm, term):
        # TODO: This is not so efficient, but the best I can do with the dict/tuple representation...
        n_known = len(term)
        n_unknown = n_known + 1
        # now to find the number of items in my_lm.ngrams[n_unknown) starting with term
        if n_unknown > 1:
            return sum([k[:-1] == term for k in my_lm.ngrams[n_unknown].keys()])
        else: # n_unknown is 1, no history given, so return count of all unigram extentions
            return len(my_lm.ngrams[1].keys())

    exts = [num_extensions(my_lm, history[x:]) for x in range(1, order+1)]
    one_minus_lambdas = [exts[x] / (exts[x] + my_lm.ngrams[order]) for x in range(1, order+1)]

    lambdas = [1 - y for y in one_minus_lambdas]

    return lambdas

def kneser_ney


def none_eqn(*args):
    return []









