__author__ = 'Kristy'
import math


####################SCORE FUNCTIONS ###############################

def ioscorefunct1(cluster_object, i, j, ):
    """Implement the agglommerative clustering score with tf.idf-like effect and size normalisation."""
    '''                     Nij            1
    sum(w in Ai and Aj)[    ----    x  --------    ]
                            |Aw|       |Ai||Aj|
    '''
    def normaliser(x, y):
        return math.sqrt((x+y)/(x*y))
    def documentcount(n):
        return len(cluster_object.give_docs_in_cluster(n, only_ins=True))

    #unique word count in i and in j (this is interpreted as tokens)
    wc_i = len(cluster_object.cluster_lms[i].ngrams[1].keys())
    wc_j = len(cluster_object.cluster_lms[j].ngrams[1].keys())

    S = sum([normaliser(documentcount(i), documentcount(j)) /
        (cluster_object.global_lm.docfreq[1][word] * wc_i * wc_j)
        for word in cluster_object.cluster_lms[i].ngrams[1].keys() & cluster_object.cluster_lms[j].ngrams[1].keys()])
    return S

# below not needed, it was hard-coded
# def minimize_unigram_perplexity(cluster_object, i, j):
#     """Highest score is the combination that gives the smallest increase in perplexity.
#     Ie Maximise the value of the probability of a sentence (as unigrams) given the topic.
#     I is the established cluster, j is the new sentence."""
#
#     # get lm of established thing i, apply it to sentence j
#     # TODO: Is J a sentence or a cluster object? If latter, get j.data
#     return cluster_object.cluster_lms[i].give_sentence_prob(j)
#



################### CLUSTER PARAMETERS ###########################

#dict of what cluster types mean what params here
class ClusterBase(object):
    def __init__(self, *args):
        '''Initialise cluster parameters from a cluster name.'''
        self.extraargs = args
        self.ngram_order = 1
        self.lm = ''
        self.smoothing = ''

        self.lm_names = () # Tuple of names that define what LM combination used for this clustering

    def __repr__(self):
        return str(dir(self))


class SoftClusterParams(ClusterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft = True


class IyerOstendorf2(SoftClusterParams):
    """Implement soft-clustering to create LMs for each topic using EM.
    Can have either random or topic-based hard-cluster initialisation."""
    def __init__(self, *args, **kwargs):
        SoftClusterParams.__init__(self, *args, **kwargs)
        self.scorefunction = None #TODO
        #self.ngram_order = 1
        self.lm_names = ('add-one', 'none', 'none')


class HardClusterParams(ClusterBase):
    def __init__(self, *args):
        super().__init__()
        self.soft = False
    def __str__(self):
        return 'Hard Cluster'

class IyerOstendorf1(HardClusterParams):
    """Parameters for the first step of I+O clustering."""
    def __init__(self, *args):
        '''Make cluster parameters item'''
        HardClusterParams.__init__(self, args)
        self.scorefunction = ioscorefunct1
        #print(self.scorefunction)
        self.ngram_order = 1
        self.iters = False
        self.lm_names = ('add-one', 'none', 'none')

class Goodman(HardClusterParams):
    """Parameters for Goodman's clustering sentences into topic clusters"""
    def __init__(self, *args):
        raise NotImplementedError
        HardClusterParams.__init__(self, args)
        #self.scorefunction = minimize_unigram_perplexity
        self.ngram_order = 1
        self.iters = 2 # TODO: Set iters better?
        self.scorefunction = None
        self.lm_names = ('add-one', 'none', 'none')


    def __str__(self):
        return 'Parameters for a Iyer and Ostendorf(agglommerative) clustering method.'

if __name__=='__main__':
    a = IyerOstendorf1()
    print(a)
    b = IyerOstendorf2()
    print(b)