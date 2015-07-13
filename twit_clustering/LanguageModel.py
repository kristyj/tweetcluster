__author__ = 'Kristy'

from collections import Counter, deque, defaultdict
import itertools
from twit_clustering.LMParams import LMParams
import numpy as np
import math

class LanguageModel(object):
    """Language model class. This keeps raw ngram counts, and gives the probabiity based on the smoothing used.
    Can print to file??"""
    startsymbol="<s>"; endsymbol="</s>"

    def __init__(self, documents, n, lm, smoothing, discparams, holdbuild=False, bydoc=False, givedf=False):
        """Build the language model based on the document.
        Use holdbuild to delay the building process til later."""
        #print(documents)
        if type(documents) != list: #document is filename
            print('Make sure the first argument is the input.\n'
                  'Else you are building a LM on a string or a file type. \n'
                  'Please pre-open the file and break into words as a list.\n')
            raise TypeError
        else:
            self.lm_params = LMParams(n, [lm, smoothing, discparams]) #now lm_params holds all the info about things that need to be trained
            self.input = documents
            print("Now setting LM order")
            self.order = self.lm_params.order
            self.bydoc = bydoc
            self.give_df = givedf


        if not holdbuild:
            self.build_ngrams(self.input, self.order)

    def make_flat_iterator(self, doclist):
        """Make an iterator over each word of the input including start/end symbols as needed"""
        print("Creating a flat iterator")
        start = [self.startsymbol]* self.order
        end = [self.endsymbol]*self.order
        if type(doclist[0])== list: #if list is more than 1d
            return itertools.chain.from_iterable([start + x + end for x in doclist])
        else:
            return itertools.chain(start + doclist + end ) #1d list gives word by word

    def make_2d_iterator(self, doclist):
        #TODO: Untested. Should return the number of iterators needed
        start = [self.startsymbol]* self.order
        end = [self.endsymbol]*self.order
        return [itertools.chain(start + x + end ) for x in doclist ]


    @classmethod
    def duplicate_to_ngrams(cls,  sentencelist, n, delete_sss=True):  #works
        '''Take a list of text, turn into a list of n-grams.
        delete_sss excludes tokens that are only start/end of sentence symbols.'''
        def make_tup(x,m):
            return tuple(sentencelist[x : x+m])
        def check_not_end(x): #works
            if x.count(LanguageModel.startsymbol) == len(x) or x.count(LanguageModel.endsymbol) == len(x):
                return False #give false if all sss
            else:
                return True #give true if its not sss
        if delete_sss:
            #print(range(len(sentencelist[n:])+1))
            a = [tuple(sentencelist[x : x+n]) for x in range(len(sentencelist[n:])+1)]
            return [y for y in a if check_not_end(y)==True]
        else:
            return [tuple(sentencelist[x : x+n]) for x in range(len(sentencelist[n-1:]))]


    def build_ngrams(self, text_by_line, n, bydoc=False):
        """Construct self.ngrams based on text counts and parameters in self.lm_parameters.
        Create self.ngrams, self.totalwords, self.lines, self.docfreq, (self.lm_by_doc) """

        print(self.lm_params.__dir__())

        # initialise variables, holders
        self.totalwords, self.lines = 0, 0
        lastnwords = deque(maxlen=n) #initialise deque

        # build all orders if specified, else just n and n-1 (for history)
        range_to_build = (range(n, 0, -1) if self.lm_params.save_lower_orders else range(n, n-2, -1))

        for x in range_to_build:
            print(x)

        self.df = {}
        self.lm_by_doc = []

        # light processing to remove exclusively start/end symbols
        from itertools import permutations
        removeesymbols = set(item for sublist in [permutations([LanguageModel.startsymbol]*self.order + [LanguageModel.endsymbol]* self.order, y) for y in range(self.order)] for item in sublist)
        print(removeesymbols)

        # removeesymbols1 = set(tuple([LanguageModel.startsymbol] * x) for x in range(1,n+1))
        # removeesymbols2 = set(tuple([LanguageModel.endsymbol] * x) for x in range(1,n+1))
        # removeesymbols = removeesymbols1.union(removeesymbols2)

        self.ngrams = {m :  Counter() for m in range_to_build}  #  build descending ngram dicts
        self.docfreq = {m: Counter() for m in range_to_build}

        # iterate over every doc/word
        for doc in  text_by_line :
            self.lines += 1
            thisdoc = {m: Counter() for m in range_to_build}
            for word in [self.startsymbol] * (self.order-1) + doc + [self.endsymbol] * (self.order-1):
                # for each word
                self.totalwords += 1
                lastnwords.extend([word])
                for m in range_to_build:        # do biggest possible ngram first
                    #print("Building this number of words", m, 'starting', n-m)
                    if m <= len(lastnwords):    # exclude where m does not get enough data eg start of text

                        # increment count for order, word combination
                        mywords = list(itertools.islice(lastnwords, n-m, None)) # deque slicing

                        #mywords = tuple(lastnwords[(n-m):])
                        print("mywords",mywords)
                        if tuple(mywords) not in removeesymbols: # ignore exclusively start/end symbols
                            thisdoc[m][tuple(mywords)] += 1

            # If DF and by_document counts are required, update these
            if bydoc:
                self.lm_by_doc.append(thisdoc)
            if self.give_df:
                for m in range_to_build:
                    for word in thisdoc[m].keys():
                        self.docfreq[m][word] += 1

            # update the global LM based on this document
            for m in range_to_build:
                self.ngrams[m].update(thisdoc[m])

    #
    #
    #
    #
    #
    # def build_ngram_counts(self, textinput_iterator, n,
    #                        buildlowerorder=self.lm.save_lower_orders,
    #                        givelines = True,
    #                        givedf=self.bydoc,
    #                        , givelines=False, givedf=False):
    #     #TODO: May contain extra start/end tokens
    #     """Put together dictionaries of ngrams and how often they occur, where n is the order.
    #     if buildlowerorder then self.ngrams[1], self.ngrams[2] etc constructed, not just self.ngrams[n].
    #     if givelines returned, return the number of documents, else 0
    #     if givedf chosen, then document frequency is calculated, else a blank dictionary returned.
    #     """
    #
    #     print("Building ngram counts now.")
    #     totalwords = 0
    #     lines = 0; df = {}
    #     lastnwords = deque(maxlen=n) #initialise deque
    #     range_to_build = (range(n, 0, -1) if buildlowerorder else range(n, n-2, -1))
    #
    #     print(removeesymbols)
    #
    #     #if buildlower is false, only builds the current n order and n-1 order (histories)
    #
    #     if givedf == False: #efficent, but takes corpus as one document
    #         for word in textinput_iterator:
    #             totalwords +=1
    #             lastnwords.extend([word])
    #             for m in range_to_build: #do biggest possible ngram first
    #                 if m >= len(lastnwords): #cannot capture all orders at eg start of text
    #                     # increment count for order, word combination
    #                     mywords = list(itertools.islice(lastnwords, n-m, n))[0] #deque slicing
    #                     #mywords = tuple(lastnwords[(n-m):])
    #                     print("mywords",mywords)
    #                     myCounters[m][tuple(mywords)] += 1
    #         myCounters[0] ={(): totalwords}
    #         if givelines is False and givedf is False:
    #             return myCounters
    #         # TODO: This is not an elegant solution to return variable args
    #         else:
    #             return itertools.compress([lines, df, myCounters], [0, 0, 1])  # return only myCounters
    #
    #     else: # returns extra info about document frequency, lines in corpus.
    #         #only take this option if a 2d iterator is created.
    #         myDocFreq = {m:Counter() for m in range_to_build}
    #         prevword = ''
    #         for word in textinput_iterator:
    #             #print("This should be a word,", word)
    #             totalwords +=1
    #             if prevword != LanguageModel.startsymbol and word == LanguageModel.startsymbol:
    #                 lastnwords.clear(); donotadd = []
    #                 lines +=1
    #             lastnwords.extend([word])
    #             #print(lastnwords)
    #             for m in range_to_build:
    #                    # print("m", m)
    #                     cn = len(lastnwords)
    #                     if m <= cn:
    #                         these_words = tuple(itertools.islice(lastnwords, cn-m, cn+1)) #list(itertools.islice(q, 3, 7))
    #                         #print("thesewords",m, these_words)
    #                         if these_words not in removeesymbols:
    #                             myCounters[m][these_words] +=1
    #                             if these_words not in donotadd:
    #                                 myDocFreq[m][these_words] +=1
    #                                 donotadd.append(these_words)
    #                         else:
    #                             pass
    #                             #print("word in removesymbols")
    #             prevword = word
    #         myCounters[0] = {(): totalwords}
    #         return itertools.compress([lines, myDocFreq, myCounters], [givelines, givedf, 1])
    #
    #
    #         # for line in textinput_iterator: #for each line
    #         #     lastnwords.clear()
    #         #     donotadd = []
    #         #     lines +=1
    #         #     for word in line:
    #         #         totalwords +=1
    #         #         lastnwords.extend(word)
    #         #         for m in range_to_build:
    #         #             if m >= len(lastnwords):
    #         #                 these_words = tuple(itertools.islice(lastnwords, n-m, len(lastnwords)+1)) #list(itertools.islice(q, 3, 7))
    #         #                 myCounters[m][these_words] +=1
    #         #                 if these_words not in donotadd:
    #         #                     myDocFreq[m][these_words] +=1
    #         #                     donotadd.append(these_words)
    #         # myCounters[0] = {(): totalwords}
    #         # return itertools.compress([lines, myDocFreq, myCounters], [givelines, givedf, 1])


    def build_count_of_counts(self):
        out_dict = {}
        for i, tupcount in self.ngrams.items():
            #print("tupcount",i,tupcount)
            if not self.lm_params.save_lower_orders and i!= max(self.ngrams.keys()):
                #print(i)
                continue
            out_dict[i] = Counter()
            for tup, freq in tupcount.items():
                out_dict[i][freq] +=1
        self.count_of_counts = out_dict
        return out_dict

    @staticmethod
    def calc_total_wc(ngram_dict):
        return {i: sum(list(ngram_dict[i].values()))for i in ngram_dict.keys()}


    def calc_vocab_size(self):
        self.vocab_size = {i: len(self.ngrams[i].keys()) for i in self.ngrams.keys()}
        return self.vocab_size


    def give_gram_prob(self, input_tuple, format=1):
        input_tuple = tuple(input_tuple)
        tuple_counts = [self.ngrams[i].get(input_tuple, 0) for i in range(1, self.order + 1) if i in self.ngrams.keys()]
        #TODO: feed tuple counts, count_of_counts, wc, vocabsize into self.lmparams.lm_eqn etc and return probabiity
        #print(self.lm_params.lm_eqn)
        p = self.lm_params.lm_eqn(self, self.order, input_tuple) #regular probability
        #print("p",p)
        if format == 1:
            return p
        elif format == 2:
            return -math.log2(p)
        elif format == 10:
            return -math.log10(p)

    def give_sentence_prob(self, sentence_list, format=1):
        if format==1:
            product = 1
            for tup in LanguageModel.duplicate_to_ngrams(sentence_list, self.order):
                product = product * self.give_gram_prob(tup, format=format)
            return product
        else: # format == 2 or 10
            return sum(self.give_gram_prob(tup, format=format) for tup in LanguageModel.duplicate_to_ngrams(sentence_list, self.order))


    def give_perplexity(self, sentence_list):
        return math.pow(- 1/len(sentence_list) * self.give_sentence_prob(format=2), 2)


#
# class GlobalLanguageModel(LanguageModel):
#     '''Build a LM over all the documents in the input.'''
#     def __init__(self, *args):
#         LanguageModel.__init__(self, *args)
#         #build a global counts over entire corpus for global stats
#         print("we got here")
#         self.lines, self.docfreq, self.ngrams = self.build_ngram_counts(self.make_flat_iterator(self.input), self.order,
#                                               #buildlowerorder=self.lm_params.save_lower_orders,
#                                               givedf=True, givelines=True)
#         #each of these reiterates of the self.ngrams dict
#         self.count_of_counts = self.build_count_of_counts() #self.ngrams#buildlowerorder = self.lm_params.save_lower_orders)
#         self.wc = self.calc_total_wc(self.ngrams)
#         self.vocabsize = self.calc_vocab_size(self.ngrams)


    def increment_counts(self, sentence_list):
        n = self.order
        range_to_build = (range(n, 0, -1) if self.lm_params.save_lower_orders else range(n, n-2, -1))
        for order in range_to_build:
            for tup in LanguageModel.duplicate_to_ngrams(sentence_list, n):
                self.ngrams[order][tup] +=1 #Type is counter so default value 0 assumed

                if order == self.order: # one token per new word
                    self.totalwords +=1

        # TODO: Below are probably time-wasting
        self.vocab_size = self.calc_vocab_size()
        self.count_of_counts = self.build_count_of_counts()


    def make_tdf_matrix(self, maxcommonvocab):
        """Use the by-document language models in self.lm_by_doc and  transfer into matrix"""

        #find vocabulary and stopwords
        stopwords = self.ngrams[self.order].most_common(50) #TODO: See in internet where to put stopword cutoff
        restricted_vocab, their_counts = self.ngrams[self.order].most_common(maxcommonvocab+len(stopwords))
        self.vocab = [w for w in restricted_vocab if w not in stopwords][:maxcommonvocab]

        #cleanup
        del restricted_vocab; del their_counts; del self.ngrams
        #build an index for each document over the defined vocabulary
        self.tdm = np.zeros((len(self.vocab),self.lines)); row_id=0
        for doc_model in self.lm_by_doc:
            #self.doc_ids.append(id)
            self.tdm[row_id] = [doc_model.give_gram_prob(word) for word in restricted_vocab]
        print("Term document matrix initialised, it looks like this! ", self.tdm)




if __name__=="__main__":
    #testing - example sentences
    sentences = [['This','is','an','example','sentence'],["this","is","an","example", "blackbird"]]

    # build the LM
    mylm = LanguageModel(sentences, 3, 'maximum-likelihood', 'none', 'none', holdbuild=False, bydoc=True, givedf=True)
    #args are: self, documents, n, lm, smoothing, discparams, holdbuild=False, bydoc=False, give_df=False):

    # inspect the global language model
    print("General count dictionary:", mylm.ngrams) # note that lower orders are saved based on LM combination
    print('count_of_counts', mylm.build_count_of_counts())
    print("Docfreq", mylm.docfreq)
    print("Vocabsize:", mylm.calc_vocab_size())

    a = mylm.give_gram_prob(["an", "example", "sentence"])
    b = mylm.give_gram_prob(["is", "an", "foo"])
    c = mylm.give_sentence_prob(["This","is","an","example","blackbird",])

    print(a, b, c)




#
# class TermDocumentMatrix(LanguageModel):
#     '''Create a LM over each individual document, ability to give a term-document matrix'''
#     def __init__(self, documents, order, smoothing=None, holdbuild=False, maxcommonvocab= 1000, cutmostfrequent=50):
#
#         #self.input, order, smoothing as lmparams, each_line_doc set
#         LanguageModel.__init__(documents, order, smoothing, eachlineisdoc=True, holdbuild=True)
#
#         #build a global counts over entire corpus for global stats
#         self.ngrams = self.build_ngram_counts(self.make_flat_iterator(self.input), self.order,
#                                               buildlowerorder=self.lm.build_all_orders)
#         #choose the vocabulary for the term-document matrix
#         stopwords = self.ngrams[self.order].most_common(50) #TODO: See in internet stopword cutoff
#         restricted_vocab, their_counts = self.ngrams[self.order].most_common(maxcommonvocab+len(stopwords))
#         self.vocab = [w for w in restricted_vocab if w not in stopwords][:maxcommonvocab]
#         del restricted_vocab; del their_counts
#
#         #build document LMs over the vocabulary
#         doclms = [self.build_ngram_counts(self.make_flat_iterator(doc), self.order, buildlowerorder=False)for doc in self.input]
#
#         # put into matrix
#         #TODO as yet no smoothing, no tfidf
#         self.tdm = [[doclms[self.order][word] for word in restricted_vocab] for doc in doclms]
#
#
#
#offcuts

    # def flatten_docs(self):
    #     ''''Make a nested list of docs and words into a flat list with start and end symbols'''
    #     return [line for line in [[self.startsymbol]*self.order + d + [self.endsymbol]* self.order for d in self.input]]

# class LMParams():
#     '''Keep track of information about different types of LMs, just storing attributes'''
#     def __init__(self, name):
#         if name=='':
#             pass
#         elif name=='':
#             pass
#         else:
#             self.build_all_orders = False
#lastnwords.extend()
        # myCounters = {m :  Counter( #this is the previous implementation
        #     [tuple(textinput[x: x+m]) for x in range(len(textinput[m:])+1)]
        # ) for m in range(n if buildlowerorder else 1)} #build descending ngram models



#from __init__
            #
            #
            # #construct blank dictionaries to score info in
            # self.ngrams = {}; self.probs = {} #all need these
            # self.countofcounts = {}; self.totalcounts = {}; self.vocabsize = {};
            # #add prefix for the order of the model [this would be more efficient without duplication, eg as tries]
            # if self.lm_params.save_lower_orders:
            #     for i in range(1,self.order +1):
            #         self.ngrams[i] = {} #key is tuple of words, value count of occurrences
            #         self.countofcounts[i] = {} #key is number of occurrences, value is no of words that occur this often
            #         self.document_frequencies[i] = {} #key is tuple of words, value is no of docs they occur in
            #         self.totalcounts[i] = {}  #TODO: Why is this here?
            #         self.vocabsize[i] = int #counts total tokens over whole corpus
            #         self.probs[i] = {} #key is tuple of words, value the probability #TODO: log/linear?
            # else:
            # self.ngrams = {}; self.countofcounts = {}; self.totalcounts = {}; self.vocabsize = {}; self.probs = {}
            # if holdbuild: #used for inheriting classes that construct different things
            #
            #
            # else: #build the standard lm over all the documents
            #     #self.doc_list = [self.startsymbol*self.order + d + self.endsymbol* self.order for d in self.input]
            #
            #     self.ngrams = self.build_ngram_counts(self.make_flat_iterator(self.input), self.order, buildlowerorder=True)
            #     self.totalcounts = {i: sum(list(self.ngrams[i].values))for i in self.ngrams.keys()}
            #     self.vocabsize = {i: len(self.ngrams[i].keys()) for i in self.ngrams.keys()}
            #
            #     self.probs = self.counts_to_smoothed_prob()

#
# class ByDocumentLanguageModel(LanguageModel):
#     '''Build LMs over each document given separately.'''
#     def __int__(self, *args, maxcommonvocab = 1000):
#         LanguageModel.__init__(self, *args)
#
#         #handle a global model for setting parameters
#         global_model = GlobalLanguageModel(*args)
#         self.doc_count, self.doc_frequency, global_model.ngrams = global_model.build_ngram_counts(
#             self.make_2d_iterator(global_model.input.giveWholeCorpus), givelines=True, givedf=True)
#         self.input.lines = self.doc_count
#
#         #find vocabulary and stopwords
#         stopwords = global_model.ngrams[self.order].most_common(50) #TODO: See in internet where to put stopword cutoff
#         restricted_vocab, their_counts = global_model.ngrams[self.order].most_common(maxcommonvocab+len(stopwords))
#         self.vocab = [w for w in restricted_vocab if w not in stopwords][:maxcommonvocab]
#
#         #cleanup
#         del restricted_vocab; del their_counts; del global_model
#         self.input.seekFileStart()
#
#         #build an index for each document over the defined vocabulary
#         self.tdm = np.zeros((len(self.vocab),self.lines)); row_id=0
#         self.doc_ids = []
#
#         while self.input.eof == False:
#             id, document = self.input.giveTextSplit(giveId=True) #get the next tweet and id
#             self.doc_ids.append(id)
#             doc_model = GlobalLanguageModel(document) #TODO: reset the params so that there is only the necessary order build
#             self.tdm[row_id] = [doc_model.give_gram_prob(word) for word in restricted_vocab]
#         print("Term document matrix initialised, it looks like this! ", self.tdm)
