__author__ = 'Kristy'

from twit_clustering.ComparisonTable import ComparisonTable
import twit_clustering.ClusterParams as Cp
from twit_clustering.LanguageModel import LanguageModel
from twit_clustering.DocSplitter import DocSplitter
import numpy as np
import random


class ClusterContainer():
    """An object that handles which documents/objects are in clusters.
    Handles the initialisation, and merging.
    Flat vs agglomerative, hard/soft are handled by child classes.
    Responsible for writing eventual cluster membership to file.
    Uses comparisontable and documentsplitter, also LM."""

    def __init__(self, data_in_list, m, clusterparamclass, fieldnames=(), outdirectory='./clusters/', prefix=None):
        """Initialise using data in list format where text is split into list
        (can have multiple fields too, if fieldnames is filled, text is last if not declared),
        with the highest level being each document.
        clusterparams class is a class object, see ClusterParams.py for information.
        m is Final number of clusters wanted,
        outdirectory and prefix refer to write information."""
        if type(data_in_list) != list:
            print("The data looks like this,", data_in_list)
            print("Please give the cluster container data in list format.")
            exit()
        elif len(fieldnames) > 0 and len(fieldnames) != len(data_in_list[0]):
            print("Your label length does not match the data. Exiting.")
            exit()
        else:
            print("Initialising over data", data_in_list)

            self.m = m; self.fieldnames = fieldnames
            self.data = data_in_list #these are now identifiers
            self.num_docs = len(self.data)
            self.cps = clusterparamclass()
            self.soft = self.cps.soft
            self.out_dir = outdirectory
            self.file_prefix = prefix
            self.lm_order = self.cps.ngram_order

            try:
                self.text_index = [1 if x.lower().startswith('text') else 0 for x in fieldnames].index(1)
            except ValueError:
                self.text_index = -1

        #basic data storage:
            #store 1/0 whether each item is in the cluster (or % if soft-clusters)
                #list same length as data

            #eg self.doc_cluster_asm[5][3] is cluster 3 doc number 5's assignment to it.
            self.doc_cluster_asm = np.zeros((self.num_docs, self.m))

            #store a LM for each cluster, at init blank #in dictionary for access not dependent on size
            self.cluster_lms = {i: LanguageModel for i in range(self.num_docs)}

            #store where the similarity can  be computed
            self.cluster_similarity = ComparisonTable(self.num_docs, self.cps.scorefunction)


    #define soft/hard function variants within the __init__
    #hard varinant
        def hard_give_clusters_for_doc(newself, doc_id, just_best_score=False):
            if just_best_score:
                return np.where(newself.doc_cluster_asm[doc_id] == 1)
            else:
                return newself.doc_cluster_asm[doc_id]
    #hard variant
        def hard_give_docs_in_cluster(newself, cluster_id, only_ins=False):
            if only_ins:
                return np.where(newself.doc_cluster_asm[:, cluster_id]==1)
            else:
                return newself.doc_cluster_asm[:, cluster_id]
        #soft variant
        def soft_give_clusters_for_doc(newself, doc_id, just_best_score=False):
            '''Return the vector of values for the doc (over all clusters), or if just_best_score the maximum cluster id.'''
            if just_best_score:
                return np.argmax(newself.doc_cluster_asm[doc_id])
            else:
                return newself.doc_cluster_asm[doc_id]
    #soft variant
        def soft_give_docs_in_cluster(newself, cluster_id, just_best_score=False):
            '''Return the document scores for the cluster, or if just_best_score then the docs that are maximally assigned here.'''
            if just_best_score:
                return np.where(newself.soft_to_hard_clusters()[:,cluster_id]==1)
            else:
                return newself.doc_cluster_asm[:, cluster_id]

        # def give_clusters_for_doc(*args, **kwargs):
        #     '''Choose which display function used'''
        #     if self.soft:
        #         return soft_give_clusters_for_doc(*args, **kwargs)
        #     else:
        #         return hard_give_clusters_for_doc(*args, **kwargs)
        #
        # def give_docs_in_cluster(*args, **kwargs):
        #     '''Choose which display function used.'''
        #     if self.soft:
        #         return soft_give_clusters_for_doc(self, *args, **kwargs)
        #     else:
        #         return hard_give_clusters_for_doc(self, *args, **kwargs)


        if self.soft:
            self.give_clusters_for_doc, self.give_docs_in_cluster = soft_give_clusters_for_doc, soft_give_docs_in_cluster
        else:
            self.give_clusters_for_doc, self.give_docs_in_cluster = hard_give_clusters_for_doc, hard_give_docs_in_cluster

#soft variant other functions
    def soft_to_hard_clusters(self):
        '''Return a matrix from soft clusters that gives the best assignment as they stand.'''
        if self.soft:
            return np.array([np.where(a==max(a), 1, 0) for a in self.doc_cluster_asm])
        else:
            print("Only hard clusters present, so returning those.")
            return self.doc_cluster_asm

    def merge_two_clusts(self, l, r):
        """Put docs from cluster number r in l. Recalculate LMs"""
        self.doc_cluster_asm[:, l] = np.add(self.doc_cluster_asm[:, r], self.doc_cluster_asm[:, l])
        self.doc_cluster_asm[:, r] = np.zeros(np.shape(self.doc_cluster_asm[r]))

    def add_new_to_clust(self, l, new_id):
        """Hard assign the new document to cluster l"""
        self.doc_cluster_asm[new_id, l] = 1

    def __str__(self):
        return "Cluster Container (parent)"

    def write_to_files(self):
        """Write the document ids for the things in the cluster to the specified filename (m files created)."""
        for n in range(self.m):
            o = open(self.out_dir+self.file_prefix+'_'+str(n)+'.docs', 'w', encoding='utf-8')
            docids = self.give_docs_in_cluster(n, only_ins=True)
            for docid in docids:
                o.write('\t'.join(self.data[docid])+'\n')

    def print_to_console(self):
        """Print the document ids for each cluster to the terminal."""
        for n in range(self.m):
            docids = self.give_docs_in_cluster(n, only_ins=True)
            for docid in docids:
                print('\t'.join(self.data[docid]))


    def initialise_randomly(self, equalsize = True):
        """Initialise random assingments of documents to clusters.
        Choose equalsize if hard assignments are made and each cluster should have an equal number at starting.
        Else each doc is randomly assigned to any of the clusters."""

        if equalsize and self.soft == False: #only for hard clustering
            # number docs that each cluster gets
            equaldist = [self.num_docs // self.m for x in range(self.m)]

            # number of docs assigned to random clusters (less than m)
            leftovers = [1 if x < self.num_docs % self.m else 0 for x in range(self.m)]
            leftovers = random.shuffle(leftovers)

            # number docs in each cluster (base + leftovers)
            clustersize = np.add(equaldist + leftovers)
            temp_docids = list(range(self.num_docs))

            for clustid, numdocs in enumerate(clustersize):
                chosen_docs = random.sample(temp_docids, numdocs)
                for x in chosen_docs:
                    self.doc_cluster_asm.all[clustid] = 1
                    del temp_docids[x]

        elif equalsize and self.soft:
            print("Cannot perform an equal assignment for soft clustering processes.")
            raise TypeError

        else: #not equalsize,
            for docno, ass_vector in enumerate(self.doc_cluster_asm):
                if self.soft:
                    rand_vector = [random.random() for x in self.m]
                    ass_vector = [x/sum(rand_vector) for x in rand_vector]
                else:
                    ass_vector[random.randint(0,self.m-1)]= 1

                # resetting values in ass_vector (mutable) should reset vals in self.doc_cluster_asm
                # retired: self.doc_cluster_asm[docno] = ass_vector





class FlatClusterContainer(ClusterContainer):
    '''A container that handles flat clustering'''
    def __init__(self, iterations, *args):
        ClusterContainer.__init__(*args)
        self.max_iters = iterations
        self.n_initial_clusters = self.m
        self.bookmark = 0

    def initialise_normally(self):
        ct = ComparisonTable(self.m, self.cps.scorefunction)
        self.global_lm = LanguageModel(self.data, self.cps.ngram_order) #TODO: This may have to seek 0,0
        for i in range(self.m):
            self.doc_cluster_asm[i][i]=1 #add all documents to one cluster each
            docids = self.give_docs_in_cluster(self, i, only_ins=True)
            self.cluster_lms[i].build_ngram_counts([self.data.docid[self.text_index] for docid in docids], self.cluster_lms[i].order)
            #TODO: Check that LMs are using appropriate params, collecting correct fields.
            self.bookmark +=1

    def cluster(self):
        iters = 0
        while iters < self.max_iters:
            while self.bookmark < self.num_docs:
                new_doc_data = self.data[self.bookmark][1] #TODO remove hard coding for field
                scores = [self.cluster_lms[i].give_sentence_prob(new_doc_data) for i in range(self.m)]
                best_cluster = np.argmax(scores)
                self.add_new_to_clust(best_cluster, self.bookmark)
                self.cluster_lms[best_cluster].increment_counts(new_doc_data)
            self.bookmark = 0
        iters += 1

        #test next document on all the lms, save in vector
        # add to relevant cluster
        #recalculate changed LMs


class AgglommerativeClusterContainer(ClusterContainer):
    """Makes hierarchical clusters"""
    def __init__(self, *args, **kwargs):
        ClusterContainer.__init__(self, *args, **kwargs)
        self.n_clusters = self.num_docs
        self.cluster_tree = {i: (i,) for i in range(self.n_clusters)}

    def __str__(self):
        pass

    def initialise_normally(self):
        self.doc_cluster_asm = np.identity(self.num_docs)
        self.global_lm = LanguageModel(self.data, self.cps.ngram_order, self.cps.lm, self.cps.smoothing,) #TODO: This may have to seek 0,0
        lines, self.docfreq, self.global_lm.ngrams = self.global_lm.build_ngram_counts(self.data, self.lm_order, givedf=True)
        for i in range(self.m):
            docids = self.give_docs_in_cluster(self, i, only_ins=True)
            print(docids, "docids")
            # assign lm to each cluster
            self.cluster_lms[i].ngrams = self.cluster_lms[i].build_ngram_counts(
                [self.data[docid][self.text_index] for docid in docids], self.lm_order,
                )
        self.ct = ComparisonTable(self.num_docs, self.cps.scorefunction)
        self.ct.initial_populate(self) #build comparison table
        print("Comparison table initialised")


    def cluster(self):
        while self.n_clusters > self.m:
            l,r = self.ct.find_to_merge()  #find smallest score
            self.cluster_tree[l] = (self.cluster_tree[l], self.cluster_tree[r])
            del self.cluster_tree[r]
            self.merge_two_clusts(l, r)
            self.ct.set_to_zeros(l, r)
            self.ct.recalculate_when_growing(self, l,)



def main():

    #data = [[1,['This','is','an','example','poo']],[2,["this","is","an","example", "blackbird"], [3,["example"]]]]
    ids, textdata = DocSplitter('split_docs.train').giveWholeCorpus(giveId=True)
    myhierclusters = AgglommerativeClusterContainer(list(zip(ids,textdata)), 4, Cp.IyerOstendorf1, fieldnames=('id', 'text'))
    print(myhierclusters.cps)
    myhierclusters.initialise_normally()
    myhierclusters.cluster()

    #myflatclusters = FlatClusterContainer( data, 2, 1)
#data_in_list, m, cluster_type,fieldnames = (), outdirectory='./clusters/', prefix=None):
    #myhierclusters = IyerOstendorf1(data, 2,)

if __name__=="__main__":
    main()