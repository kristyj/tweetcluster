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
        return len(cluster_object.give_docs_in_cluster(cluster_object, n, only_ins=True))

    #unique word count in i and in j (this is interpreted as tokens)
    wc_i = len(cluster_object.cluster_lms[i].ngrams[1].keys())
    wc_j = len(cluster_object.cluster_lms[j].ngrams[1].keys())

    S = sum([normaliser(documentcount(i), documentcount(j)) /
        (cluster_object.global_lm.docfreq[1][word] * wc_i * wc_j)
        for word in cluster_object.cluster_lms[i].ngrams[1].keys() & cluster_object.cluster_lms[j].ngrams[1].keys()])
    return S


################### CLUSTER PARAMETERS ###########################

#dict of what cluster types mean what params here
class ClusterBase(object):
    def __init__(self, *args):
        '''Initialise cluster parameters from a cluster name.'''
        self.extraargs = args
        self.ngram_order = 1
        self.lm = ''
        self.smoothing = ''

    def __repr__(self):
        return str(dir(self))


class SoftClusterParams(ClusterBase):
    def __init__(self, *args):
        super().__init__()
        self.soft = True


class IyerOstendorf2(SoftClusterParams):
    """Implement soft-clustering to create LMs for each topic using EM.
    Can have either random or topic-based hard-cluster initialisation."""
    def __init__(self, *args):
        SoftClusterParams.__init__(self)
        self.scorefunction = None #TODO
        #self.ngram_order = 1


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
        print(self.scorefunction)
        self.ngram_order = 1

    def __str__(self):
        return 'Parameters for a Iyer and Ostendorf(agglommerative) clustering method.'

if __name__=='__main__':
    a = IyerOstendorf1()
    print(a)
    b = IyerOstendorf2()
    print(b)