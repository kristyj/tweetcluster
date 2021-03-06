__author__ = 'Kristy'

from twit_clustering.ComparisonTable import ComparisonTable
import twit_clustering.ClusterParams as Cp
from twit_clustering.LanguageModel import LanguageModel
from twit_clustering.DocSplitter import DocSplitter
import numpy as np
import random
from twit_clustering.LMParams import LMParams


class BaseClusterContainer(object):
    """Class that holds the most basic information for document topic-clustering:
    - what are the documents?
    - which documents are assigned to which topic
    - what do the language models look like for each topic?
    - how many topics are we aiming for

    and handles the methods for:
    - giving the documents
    - outputting the clustered information into files/onto screen for inspection."""

    def __init__(self, data_in_list, m, clusterparamclass, fieldnames=(), outdirectory='../clusters/', prefix=None):
        """ Handle the input parameters, create basic data structures.
         data is in a list, each item is a doc, and fields can be described by names in 'field' param.
         outdirectory and prefix refer to the location that cluster files should be stored.
         clusterparamsclass refers to a parameters object, that answers questions about certain parameters used in clustering
         based on the chosen model. Eg trigrams vs bigrams, which smoothing in the LM, how many iterations etc.
         This is to make changing the parameters easier."""

    # set global parameters
        self.m = m
        self.cps = clusterparamclass()
        self.out_dir = outdirectory
        self.file_prefix = prefix

        # set up the document data as an iterator
        # ensure correct input type
        if type(data_in_list) != list: # cannot accept string data
            print("The data looks like this,", data_in_list)
            print("Please give the cluster container data in list format.")
            exit()
        elif len(fieldnames) > 0 and len(fieldnames) != len(data_in_list[0]): # cannot handle incorrect field names
            print("Your label length does not match the data. Exiting.")
            exit()
        else:
            print("Starting a cluster container object initialising over data", data_in_list)

        # check any manually-given labelling, else presume [id, text_as_list] setup per document
        self.fieldnames = fieldnames
        print("Given fieldnames are", self.fieldnames)
        try:
            self.text_index = self.fieldnames.index('text')
        except ValueError:
            self.text_index = -1
        finally:
            print("Text is in position", self.text_index)

        try:
            self.doc_ids_index = [x.lower().startswith('id') for x in self.fieldnames].index(True)
        except ValueError:
            self.doc_ids_index = 0
        finally:
            print("IDs are in position", self.doc_ids_index)

    # set up the basic text data
        self.data = data_in_list
        self.num_docs = len(self.data)
        self.doc_ids = [x[self.doc_ids_index] for x in self.data]

    # set up data structures
        self.doc_cluster_asm = np.zeros((self.num_docs, self.m)) #eg self.doc_cluster_asm[5][3] is doc 5's assignment to cluster 3.
        print('just made matrix: m', self.m, 'asm', self.doc_cluster_asm)
        self.cluster_lms = {}

    def give_docs(self):
        """Generator over all the input data, returning only each document's text field as list."""
        i = 0
        while i < self.num_docs:
            yield self.data[i][self.text_index]
            i += 1

    def write_to_files(self, idsonly=False):
        """Write the document ids for the things in the cluster to the specified filename (m files created)."""
        for n in range(self.m):
            o = open(self.out_dir+self.file_prefix+'_'+str(n)+'.docs', 'w', encoding='utf-8')
            docinds = self.give_docs_in_cluster(n, just_best_score=True)
            for docid in docinds:
                if idsonly:
                    o.write('\t'.join(self.name_this_docs(docid))+'\n')
                else:
                    o.write(self.data[docid][self.doc_ids_index]+'\t'+' '.join(self.data[docid][self.text_index])+'\n')

    def write_wcs_to_files(self, order, fulllm=True):
        """Write topic model files with the most popular words, at order specified, set fulllm as false to get only top 1000 ngrams"""
        for n in range(self.m):
            o = open(self.out_dir+self.file_prefix+'_'+str(n)+'.topcounts', 'w', encoding='utf-8')
            lm = self.cluster_lms[n]
            print("This should be an LM object", lm)
            if type(lm) == LanguageModel:
                counts = lm.give_ordered_wcs(order, all=fulllm)
                print("count output",counts)
                o.write('\n'.join(['{}\t{}'.format(' '.join(c[0]), str(c[1])) for c in counts]))

    def print_to_console(self):
        """Print the document ids for each cluster to the terminal."""
        for n in range(self.m):
            print("CLUSTER NUMBER {} -------------------------".format(str(n)))
            docids = self.give_docs_in_cluster(n, only_ins=True)
            print("docids", docids)
            for docid in docids:
                print(self.data[docid][self.doc_ids_index]+'\t'+' '.join(self.data[docid][self.text_index]))
                   # '\t'.join(self.data[docid][self.doc_ids_index], ' '.join(self.data[docid][self.text_index])))
                    #[str(x) for i, x in enumerate(self.data[docid])]))

    def name_this_docs(self, i):
        """Give indices of the docs as they are in the self.data, returning the ids from the original posts"""
        if i < len(self.doc_ids):
            return self.doc_ids[i]
        else:
            print("Document index out of range")
            return None


class SoftClusterContainer(BaseClusterContainer):
    """Object for keeping track of soft-cluster assignments between documents and topics."""
    def __init__(self, data_in_list, m, clusterparamclass, iters=5, fieldnames=(), outdirectory='../clusters/', prefix=None):
        """Initialise topics and priors for EM, but do not perform any steps."""
        super().__init__(data_in_list, m, clusterparamclass, fieldnames=fieldnames, outdirectory=outdirectory, prefix=prefix )

        self.soft = True
        self.maxiters = iters

        #initialise value placeholders
        self.normaliser = 1
        self.doc_cluster_asm = np.zeros((self.num_docs, self.m))  # values initialised in child classes
        self.current_iter = 0
        self.iter_changes = [100]

        # each EM topic mix has a prior probability (in self.priors)
        self.priors = [] # instantiated in child class
        # a lm (in self.cluster_lms[i]) either gathered on init or made in child class

    def __str__(self):
        return "Expectation maximisation holder, the mixes are {}".format(self.priors)

    def do_em(self):
        """initialise the expectation-maximisation procedure"""
        print("Now initialising EM on {} language models".format(self.m))
        while self.maxiters > self.current_iter and self.measure_mixes_change():
            # while manually set iterations remain and the model changes significantly
            # #todo: check the definition of the stopping point by self.iter_changes
            self.current_iter += 1
            print("Now performing expectaton\nThis may take some time as all documents are re-read and evaluated over each topic LM.")
            self.expectation()
            print("Now performing maximisation, this re-evaluates every n-gram by re-reading all the documents, therefore is slow.")
            self.maximisation()
            print("The prior probability of the topics changed like this:", self.iter_changes)

    def measure_mixes_change(self):
        '''For the m topics, look at the list of how much changed in the last iteration. Return True if there was a perceptible change, False if not.'''
        # TODO: Research a relevant measure
        return sum(self.iter_changes) > 0.01

    def initialise_randomly(self):
        """Give random topic assignments to each doc, summing to 1 """
        for d in self.doc_cluster_asm:
            rand_vector = [random.random() for x in range(len(d))]
            ass_vector = [x/sum(rand_vector) for x in rand_vector]
            d = ass_vector  # exploit mutability of list values.

    def give_clusters_for_doc(self, doc_id, just_best_score=False):
        """Soft variant: Return the vector of values for the doc (over all clusters),
        or if just_best_score the maximum cluster id."""
        if just_best_score:
            return np.argmax(self.doc_cluster_asm[doc_id])
        else:
            return self.doc_cluster_asm[doc_id]

    def give_docs_in_cluster(self, cluster_id, just_best_score=False, **kwargs):
        """Soft variant: Return the document scores for the cluster,
        or if just_best_score then the docs that are maximally assigned here."""
        if just_best_score:
            return [int(x) for x in np.where(self.soft_to_hard_clusters()[:, cluster_id] == 1)[0]]
        else:
            return [int(x) for x in self.doc_cluster_asm[:, cluster_id]] # get every row (each doc)

    def soft_to_hard_clusters(self):
        """Return a matrix from soft clusters that gives the best assignment as they stand."""
        if self.soft:
            return np.array([np.where(a==max(a), 1, 0) for a in self.doc_cluster_asm])
        else:
            print("Only hard clusters present, so returning those.")
            return self.doc_cluster_asm


class IyerOstendorfEM(SoftClusterContainer):
    """Class to make topic-based Language Models using Iyer and Ostendorf's methods.
    Can be initialised from flat-clustered topics."""

    def __init__(self, *args, clustercontainer=None, **kwargs):
        """Initialise data from old clusters, make relevant topic mixes."""
        super().__init__(*args, **kwargs)

        # check type of existing mixes
        if isinstance(clustercontainer, FlatClusterContainer):
            if clustercontainer.m != self.m:
                print("Different m values for pre-made data and for soft clusters")
                exit()

            print("Initialising using pre-clustered data.")

            # initialise priors based on num docs in flat clusters
            self.priors = [1/len(clustercontainer.give_docs_in_cluster(i, only_ins=True)) for i in range(self.m)]

            # take over the pre-calculated LM counts, but change the equations
            self.cluster_lms = clustercontainer.cluster_lms
            for lm in self.cluster_lms.values():
                # set order
                lm.order = self.cps.ngram_order
                # set lm_params
                lm.lm_params = LMParams(lm.order, self.cps.lm_names) # note: order from lm_params and cps could be out of sync

                # manually off, because topic LMs work over corpus now, and no longer using tf.idf
                lm.bydoc = False
                lm.give_df = False

                # clear old info
                del lm.input

                # build ngram counts for initial e-step
                lm.build_ngrams(self.give_docs(), lm.order, bydoc=lm.bydoc)

        elif clustercontainer is None:
            print("Initialising over random splits. This is not the proper Iyer/Ostendorf approach.")

            # initialise priors equally
            self.priors = [1/self.m for x in range(self.m)]

            # make random hard clusters
            fakeflats = FlatClusterContainer(0, self.data, self.m, Cp.HardClusterParams)
            fakeflats.initialise_randomly(equalsize=False) # make random hard clusters to begin

            # build initial LMs
            for i in range(self.m):
                self.cluster_lms[i] = LanguageModel([self.data[i][self.text_index] for i in fakeflats.give_docs_in_cluster(i, only_ins=True) ], self.cps.ngram_order, self.cps.lm_names, holdbuild=False, bydoc=False) # TODO: LMparams- check

        else:
            print("You are attempting to initialise on the wrong data.")
            exit()

        self.do_em()


    def expectation(self):
        """Calculate the zij value for each tweet = that is, how likely each tweet i is to belong to topic j."""
        for i, tweet in enumerate(self.give_docs()):
            # change the doc-cluster assignment values for the e-step
            self.doc_cluster_asm[i] = self.calc_p_tweet_to_class(tweet)

    def calc_p_tweet_to_class(self, sentence):
        """Calculate the probability that the tweet belongs to each class, given prior and ngram probability"""
        # get prior prob for topic, for sentence

        sentenceprobs = [self.cluster_lms[i].give_sentence_prob(sentence) for i in range(self.m)]
        priors = self.priors

        # perform Iyer and Ostendorf eqn 5
        numerators = [sentenceprobs[x] * priors[x] for x in range(self.m)]
        #calclate the denominator
        normalizer = sum(numerators)
        zijs = [numerators[x]/normalizer for x in range(self.m)]

        return zijs

    def maximisation(self): #TODO: Ongoing
        """Adjust the word|topic probability based on the posteriors, update lm counts of each ngram
        Following eqns 7 and 8 in I+O 1999."""
        print("Now performing {}th iteration, maximisation step".format(self.current_iter))

        #reset all the relevant LM counts (these are now weighted)
        for i in range(self.m):
            self.cluster_lms[i].zero_ngram_counts()

        #make a new weighted count dictionary [interims] for each bigram including the sentence weight (zij) in the topic
        print("Recounting n-grams to include weight from expectation step.")

        for did, doc in enumerate(self.give_docs()):
            # for every topic
            for t in range(self.m):
                lm = self.cluster_lms[t]
                # for every order
                for r in lm.range_to_build:
                    # for each ngram in the sentence
                    for ng in lm.duplicate_to_ngrams(doc, r):
                        # get a weighted count that can then be used for MLE and witten-bell smoothing
                        # from Iyer and Ostendorf 1999 eqn 8 (numerator)
                        lm.ngrams[r][ng] += self.doc_cluster_asm[did][t]  # ie += zij per occurrence

        # TODO: Now MLE counts are updated, the Witten-bell smoothing params can be recalculated and need to be used when calculating probability in the next e-step
        # TODO: Check that LM parameters are being passed, and that these smoothing params are being used and calculated.


        # checking if the topic priors have changed enough to continue iterating
        total_assm_to_topic = np.sum(self.doc_cluster_asm, axis=0)
        self.iter_changes = np.subtract(total_assm_to_topic, self.priors)
        total_assms = sum(total_assm_to_topic)

        self.priors = [total_assm_to_topic[t]/total_assms for t in range(self.m)]
        # NOT DOING TEMP_TOTAL_ZIJ AS CALCULATING PROBABILITY ON THE FLY
        # NOT DOING AS SIMPLY OVERWRITING DOC_CLUSTER_ASM: get temp_total_zij as sum([sum(mytopic.posteriors) for mytopic in self.topics])



class GildeaHoffmanEM(SoftClusterContainer):
    """Class to make topic-based Language Models using Gildea and Hoffmans's methods.
    Initialised only over raw data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #make a language model over all the documents recording wordcounts
        self.universal_lm = LanguageModel([t for t in self.give_docs()], 1, ('add-one', 'none', 'none'), bydoc=True, ) #TODO: other args


        # initialise random topic/document associations
        self.initialise_randomly()

        # initialise topic priors

        # initialise topic posteriors
        for topic in self.topics:
            topic.posteriors = [{word: 0 for word in document} for document in self.universal_lm.tdm]#TODO initialise some list with the universal_lm_dimensions
            print(topic.posteriors[0])

# # TODO: Initialise topics based on self.cps
        self.do_em()

    def expectation(self):
        """E-step as defined by Gildea and Hoffman"""
        pass

    def maximisation(self):
        """M-step as defined by Gildea and Hoffman"""
        pass


class HardClusterContainer(BaseClusterContainer):
    """An object with methods relevant for creating hard clusters. """

    def __init__(self, data_in_list, m, clusterparamclass, fieldnames=(), outdirectory='../clusters/', prefix=None):
        """Initialise using data in list format where text is split into list
        (can have multiple fields too, if fieldnames is filled, text is last if not declared),
        with the highest level being each document.
        clusterparams class is a class object, see ClusterParams.py for information.
        m is Final number of clusters wanted,
        outdirectory and prefix refer to write information."""

        super().__init__(data_in_list, m, clusterparamclass, fieldnames=fieldnames, outdirectory=outdirectory, prefix=prefix)

        print("data vs m", len(self.data), self.m)
        self.soft = False
        self.lm_order = self.cps.ngram_order
        self.merged_away = []

    def give_docs_in_cluster(self, cluster_id, only_ins=False, **kwargs):
        """Hard variant: Return either the vector for the cluster number,
        or if only_ins the indices (eg doc ids) as list where they were in."""
        if only_ins:
            return [int(x) for x in np.where(self.doc_cluster_asm[:, cluster_id] == 1)[0]]
        else:
            return [int(x) for x in self.doc_cluster_asm[:, cluster_id]]

    def give_clusters_for_doc(self, doc_id):
        """Hard variant: given a doc, return an integer (cluster number it is in)"""
        try:
            return int(np.where(self.doc_cluster_asm[doc_id]==1)[0][0])
        except IndexError:
            return None


    def merge_two_clusts(self, l, r):
        """Put docs from cluster number r in l. Recalculate LMs"""

        # merge the scores in the table
        self.doc_cluster_asm[:, l] = np.add(self.doc_cluster_asm[:, r], self.doc_cluster_asm[:, l])
        # actually should delete this row, because otherwise results give empty clusters
        self.doc_cluster_asm[:, r] = np.zeros(np.shape(self.doc_cluster_asm[:,r]))

        # add to list checked
        self.merged_away.append(r)

        # merge the LMs
        self.cluster_lms[l].increment_counts(self.cluster_lms[r])
        self.cluster_lms[r] = None

    def __str__(self):
        return "Hard Cluster Container (parent)"

    def initialise_randomly(self, equalsize = True):
        """Initialise random assingments of documents to clusters (returns vector for one document)
        Choose equalsize if hard assignments are made and each cluster should have an equal number at starting.
        Else each doc is randomly assigned to any of the clusters."""

        if equalsize: #only for hard clustering
            # number docs that each cluster gets
            equaldist = [self.num_docs // self.m for x in range(self.m)]

            # number of docs assigned to random clusters (less than m)
            leftovers = [1 if x < self.num_docs % self.m else 0 for x in range(self.m)]
            leftovers = random.shuffle(leftovers)

            # number docs in each cluster (base + leftovers)
            clustersize = np.add(equaldist , leftovers)
            temp_docids = list(range(self.num_docs))

            for clustid, numdocs in enumerate(clustersize):
                chosen_docs = random.sample(temp_docids, numdocs)
                for x in chosen_docs:
                    self.doc_cluster_asm.all[clustid] = 1
                    del temp_docids[x]

        else: #not equalsize,
            for docno, ass_vector in enumerate(self.doc_cluster_asm):
                ass_vector[random.randint(0,self.m-1)] = 1

                # resetting values in ass_vector (mutable) should reset vals in self.doc_cluster_asm
                # retired: self.doc_cluster_asm[docno] = ass_vector


class FlatClusterContainer(HardClusterContainer):
    '''A container that handles flat clustering'''
    def __init__(self, iterations, data_in_list, m, clusterparamclass, **kwargs):
        super().__init__(data_in_list, m, clusterparamclass, **kwargs)
        if self.m == 1:
            print("Cannot flat cluster to 1 cluster, this just returns input.")
            raise ValueError
        self.max_iters = iterations
        self.n_initial_clusters = self.m
        self.bookmark = 0

        # flat
        self.cluster_lms = {i: LanguageModel(self.data[i][self.text_index], self.cps.ngram_order, self.cps.lm_names, holdbuild=True) for i in range(self.m)}


    def initialise_normally(self):
        ct = ComparisonTable(self.m, self.cps.scorefunction)
        self.global_lm = LanguageModel([x[self.text_index] for x in self.data], self.cps.ngram_order, self.cps.lm_names) #TODO: This may have to seek 0,0
        for i in range(self.m):
            self.doc_cluster_asm[i][i]=1 #add all documents to one cluster each
            docids = self.give_docs_in_cluster(i, only_ins=True)
            self.cluster_lms[i].build_ngrams([self.data[docid][self.text_index] for docid in docids], self.cluster_lms[i].order)
            #TODO: Check that LMs are using appropriate params, collecting correct fields.
            self.bookmark += 1

    def cluster(self):
        iters = 0
        while iters < self.max_iters:
            print('clustering one iteration')
            while self.bookmark < self.num_docs:

                # get the next sentence
                new_doc_data = self.data[self.bookmark][self.text_index]
                #print("new_doc_data", new_doc_data)

                # remove the sentence from its current cluster
                current_cluster = self.give_clusters_for_doc(self.bookmark)
                if current_cluster is not None:
                    self.doc_cluster_asm[self.bookmark] = np.zeros(self.m)
                    self.cluster_lms[current_cluster].decrement_counts(new_doc_data)

                # calculate the best new assignment for the sentence
                scores = [self.cluster_lms[i].give_sentence_prob(new_doc_data) for i in range(self.m)]
                best_cluster = np.argmax(scores)

                # merge new clusters
                self.add_new_to_clust(best_cluster, self.bookmark)
                self.cluster_lms[best_cluster].increment_counts(new_doc_data)

                self.bookmark += 1
            self.bookmark = 0
            iters += 1

        #test next document on all the lms, save in vector
        # add to relevant cluster
        #recalculate changed LMs

    def add_new_to_clust(self, l, new_id):
        """Hard assign the new document to cluster l. Leave the old assignment as it was."""
        self.doc_cluster_asm[new_id, l] = 1

class AgglommerativeClusterContainer(HardClusterContainer):
    """Makes hierarchical clusters"""
    def __init__(self, *args, **kwargs):
        print('KWARGS',kwargs, 'ARGS',  args)
        super().__init__(*args, **kwargs)
        self.n_clusters = self.num_docs
        self.cluster_tree = {i: (i,) for i in range(self.n_clusters)}


        # agglomerative
        self.cluster_lms = {i: LanguageModel(self.data[i][self.text_index], self.cps.ngram_order, self.cps.lm_names, holdbuild=True) for i in range(self.num_docs)}

        # hard hierarchical cluster specific

        # #store where the similarity can  be computed
        # self.cluster_similarity = ComparisonTable(self.num_docs, self.cps.scorefunction)

        # build all the LMs based on original data
        print("Building LM for each document.")
        for i in range(len(self.cluster_lms)):
            self.cluster_lms[i].build_ngrams([self.cluster_lms[i].input], self.lm_order)
            print('all-ngrams init', self.cluster_lms[i].ngrams)

    def __str__(self):
        pass

    def initialise_normally(self):
        self.doc_cluster_asm = np.identity(self.num_docs)
        print('datalist',list(zip(*self.data)))
        self.global_lm = LanguageModel([x[self.text_index] for x in self.data], self.cps.ngram_order, self.cps.lm, self.cps.smoothing, givedf=True) #TODO: This may have to seek 0,0
        self.global_lm.build_ngrams( [x[self.text_index] for x in self.data], self.lm_order)
        for i in range(self.m):
            docids = self.give_docs_in_cluster(i, only_ins=True)
            print(docids, "docids")
            # # assign lm to each cluster
            # self.cluster_lms[i].ngrams = self.cluster_lms[i].build_ngrams(
            #     [self.data[docid][self.text_index] for docid in docids], self.lm_order,
            #     )
        self.ct = ComparisonTable(self.num_docs, self.cps.scorefunction)
        self.ct.initial_populate(self) #build comparison table
        print("Comparison table initialised")


    def cluster(self):
        """While there are docs left and too many clusters, calc score and merge."""
        while self.n_clusters > self.m:
            if self.m >= len(self.data):
                print("There are less documents than intented clusters")
                break
            l,r = self.ct.find_to_merge()  #find smallest score
            if l is None and r is None:
                print("weird early exit")
                break
            # update the cluster tree
            self.cluster_tree[l] = (self.cluster_tree[l], self.cluster_tree[r])
            del self.cluster_tree[r]

            # update the assignment matrix, and LMs
            self.merge_two_clusts(l, r)
            self.n_clusters -= 1

            # update the score table
            self.ct.set_to_zeros(l, r)
            self.ct.recalculate_when_growing(self, l,)

        # clean up the assignment array (move from n cols to m cols)
        self.doc_cluster_asm = np.delete(self.doc_cluster_asm, self.merged_away, axis=1)
        self.cluster_lms = [x for x in self.cluster_lms if x is not None]



def main():

    #ids, textdata = DocSplitter('split_docs.train').giveWholeCorpus(giveId=True)
    ids, textdata = DocSplitter('../docs/smaller_devset.tsv').giveWholeCorpus(giveId=True)

    #data = [[1,['This','is','an','example','poo']],[2,["this","is","an","example", "blackbird"], [3,["example"]]]]
    if False: # I+O testing

        myhierclusters = AgglommerativeClusterContainer(list(zip(ids,textdata)), 10, Cp.IyerOstendorf1, fieldnames=('id', 'text'), prefix='uni-agglom')
        print(myhierclusters.cps)
        myhierclusters.initialise_normally()
        myhierclusters.cluster()
        print(myhierclusters.doc_cluster_asm)
        print(myhierclusters.cluster_tree)
        myhierclusters.write_to_files()
        myhierclusters.write_wcs_to_files(1, fulllm=False)

    if False:
        # iterations, data, ngram-order, parameters
        myflatclusters = FlatClusterContainer(3, list(zip(ids, textdata)), 5, Cp.Goodman, prefix='unigram') # fields=('id', 'usr', 'text'))
        print(1)
        print('asm', myflatclusters.doc_cluster_asm)
        myflatclusters.initialise_normally()
        print(2)
        print('asm', myflatclusters.doc_cluster_asm)
        myflatclusters.cluster()
        print(3)
        print('asm', myflatclusters.doc_cluster_asm)
        print(4)
        print('clm type', type(myflatclusters.cluster_lms))
        for i in range(myflatclusters.m):
            print('lms', i, '\n', myflatclusters.cluster_lms[i].ngrams )
        myflatclusters.print_to_console()
        myflatclusters.write_to_files()
        myflatclusters.write_wcs_to_files(1, fulllm=False)

    if False:
        myem_1 = IyerOstendorfEM(list(zip(ids, textdata)), 5, Cp.IyerOstendorf2, iters=2, clustercontainer=myflatclusters, prefix='io2', )
        print(1)
        print('asm', myem_1.doc_cluster_asm)

        print(myem_1.soft_to_hard_clusters())
        myem_1.write_to_files()
        myem_1.write_wcs_to_files(1, fulllm=False)

    if True:
        gh = GildeaHoffmanEM(list(zip(ids, textdata)), 5, Cp.Goodman, iters=2, prefix='gh', )
        gh.write_to_files()
        gh.write_wcs_to_files(1, fulllm=False)

#data_in_list, m, cluster_type,fieldnames = (), outdirectory='./clusters/', prefix=None):
    #myhierclusters = IyerOstendorf1(data, 2,)

if __name__=="__main__":
    main()