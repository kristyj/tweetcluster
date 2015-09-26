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

    def __init__(self, documents, n, lm_names, holdbuild=False, bydoc=False, givedf=False):
        """Build the language model based on the document.
        lm_names is a len=3 list/tuple with the names of the base LM, smoothing LM and paramcalc.
        Use holdbuild to delay the building process til later."""
        #print(documents)
        if type(documents) != list: #document is filename
            print('Make sure the first argument is the input.\n'
                  'Else you are building a LM on a string or a file type. \n'
                  'Please pre-open the file and break into words as a list.\n')
            raise TypeError
        else:
            self.lm_params = LMParams(n, lm_names) #now lm_params holds all the info about things that need to be trained
            self.range_to_build = (range(n, 0, -1) if self.lm_params.dict['save_lower_orders'] else range(n, n-2, -1))
            self.input = documents
            #print("Now setting LM order")
            self.order = self.lm_params.order
            self.bydoc = bydoc
            self.give_df = givedf

        print('****', self.lm_params.dict)

        # create a list of start/end symbols to be ignored in LM
        from itertools import permutations
        self.removesymbols = set(item for sublist in [permutations([LanguageModel.startsymbol]*self.order + [LanguageModel.endsymbol]* self.order, y) for y in range(self.order)] for item in sublist)
        #print(self.removesymbols)

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

    def duplicate_to_ngrams(self, sentencelist, n, delete_sss=False):  #works
        """Take a list of text, turn into a list of n-grams.
        delete_sss excludes tokens that are only start/end of sentence symbols.
        This is used when evaluating test texts, a deque is used when training for efficiency."""
        #print("duplicates sentence list is ", sentencelist)
        if not delete_sss:
            sentencelist = [self.startsymbol] * (self.order - 1) + sentencelist + [self.endsymbol] * (self.order -1)

        # return list of ngrams for the sentence given
        #print('returning', [tuple(sentencelist[x: x+n]) for x in range(len(sentencelist)-n+1)])
        return [tuple(sentencelist[x: x+n]) for x in range(len(sentencelist)-n+1)]

    def give_ordered_wcs(self, order, all=False):
        """return a list of tuples (item, count) of the 1000 most common (if all is False) items in lm"""
        order_dict = self.ngrams.get(order, None)
        if order_dict is None:
            print("Cannot get these counts, not recorded.")
        else:
            return order_dict.most_common(len(order_dict.keys()) if all else 1000)

    def zero_ngram_counts(self):
        """For every existing ngram order, set the counts to zero."""
        for k in self.ngrams.keys():
            self.ngrams[k] = Counter()

    def build_ngrams(self, text_by_line, n, bydoc=False):
        """Construct self.ngrams based on text counts and parameters in self.lm_parameters.
        Create self.ngrams, self.totalwords, self.lines, self.docfreq, (self.lm_by_doc) """

        #print(self.lm_params.__dir__())

        # initialise variables, holders
        self.totalwords, self.lines = 0, 0
        lastnwords = deque(maxlen=n) #initialise deque

        # build all orders if specified, else just n and n-1 (for history)
        #print('LM parameters are!!!', self.lm_params.dict)


        # for x in range_to_build:
        #     print(x)

        self.df = {}
        self.lm_by_doc = []

        self.ngrams = {m:  Counter() for m in self.range_to_build}  #  build descending ngram dicts
        self.docfreq = {m: Counter() for m in self.range_to_build}

        self.deleted_symbols = {m: Counter() for m in range(1, self.order)}

        # iterate over every doc/word
        for doc in text_by_line :
            self.lines += 1
            thisdoc = {m: Counter() for m in self.range_to_build}
            for word in [self.startsymbol] * (self.order-1) + doc + [self.endsymbol] * (self.order-1):
                # for each word
                if word not in self.removesymbols:
                    self.totalwords += 1  # TODO: This does not count start/end symbols

                lastnwords.extend([word])
                #print('range to build', [x for x in range_to_build])
                for m in self.range_to_build:        # do biggest possible ngram first
                    #print("Building this number of words", m, 'starting', n-m)
                    if m <= len(lastnwords) and m > 0:    # exclude where m does not get enough data eg start of text

                        # increment count for order, word combination
                        mywords = list(itertools.islice(lastnwords, len(lastnwords)-m, None)) # deque slicing # need to take the last m words
                        # OLD: mywords = list(itertools.islice(lastnwords, n-m, None)) # deque slicing

                        #mywords = tuple(lastnwords[(n-m):])
                        #print("m",m, "n",n)
                        #print(lastnwords)
                        #print("mywords",mywords)
                        if tuple(mywords) not in self.removesymbols: # ignore exclusively start/end symbols
                            thisdoc[m][tuple(mywords)] += 1
                        else:
                            self.deleted_symbols[m][tuple(mywords)] += 1
            # If DF and by_document counts are required, update these
            if bydoc:
                self.lm_by_doc.append(thisdoc)
            if self.give_df:
                for m in self.range_to_build:
                    for word in thisdoc[m].keys():
                        self.docfreq[m][word] += 1

            # update the global LM based on this document
            for m in self.range_to_build:
                self.ngrams[m].update(thisdoc[m])

        self.calc_vocab_size()
        if self.lm_params.dict['save_count_of_counts']:
            self.build_count_of_counts()

    def build_count_of_counts(self):
        """Creat a dictionary with integers as keys and the number of items that occur with that frequency as value."""
        out_dict = {}
        for i, tupcount in self.ngrams.items():
            #print("tupcount",i,tupcount)
            if not self.lm_params.dict['save_lower_orders'] and i != max(self.ngrams.keys()):
                #print(i)
                continue
            out_dict[i] = Counter()
            for tup, freq in tupcount.items():
                out_dict[i][freq] += 1
        self.count_of_counts = out_dict
        return out_dict

    @staticmethod
    def calc_total_wc(ngram_dict):
        """Calculate the total number of grams (tokens) for each order, returning a dictionary"""
        return {i: sum(list(ngram_dict[i].values()))for i in ngram_dict.keys()}

    def calc_vocab_size(self):
        """Calculate the total number of types for each order returning a dictionary."""
        self.vocab_size = {i: len(self.ngrams[i].keys()) for i in self.ngrams.keys()}
        return self.vocab_size

    def calculate_histories_continuations(self, conts=True, hists=True,):
        """Iterate over all the n-grams and create dictionaries of how many alternative continuations/histories occur.
        Access as self.continuation_counts[n][the_history], self.histories_counts[n][the_continuation]"""
        self.continuation_counts = {m: {} for m in range(1, self.order)} # unigram to m-1

        self.histories_counts = {m: {} for m in range(2, self.order + 1)} # bigram to m

        for m in range(1, self.order + 1):
            if conts:
                if m-1 in self.continuation_counts.keys():  # if continuation should be calculated
                    self.continuation_counts[m-1] = Counter([k[:-1] for k in self.ngrams[m].keys()])
            if hists:
                if m in self.histories_counts.keys(): # if histories should be calculated
                    self.histories_counts[m-1] = Counter([k[1:] for k in self.ngrams[m].keys()])




    def count_continuations(self, some_history):
        """Given a tuple, return a number with the number of different (one-word) continuations that follow it."""
        # TODO: This is inefficient because it iterates over all values. Grr.
        found = 0
        continuations = 0
        hist_order = len(some_history)
        max_occurs = self.ngrams[hist_order].get(some_history, 0)

        while found < max_occurs:
            for k, v in self.ngrams[hist_order + 1].items():
                if k[:-1] == some_history:
                    found += v
                    continuations +=1

        return continuations

    def count_histories(self, some_continuation):
        """Given some word/tuple, count how many different (one-word) histories come before it."""
        # TODO: This is inefficient because of the excessive iterations (one over all ngrams for each call)
        found = 0
        histories = 0
        cont_order = len(some_continuation)
        max_occurs = self.ngrams[cont_order].get(some_continuation, 0)

        while found < max_occurs:
            for k, v in self.ngrams[cont_order + 1].items():
                if k[1:] == some_continuation:
                    found += v
                    histories += 1
        return histories


    def give_gram_prob(self, input_tuple, format=1):
        """Give the probability of the ngram, either linear or log, depending on format."""
        input_tuple = tuple(input_tuple)

        p = self.lm_params.dict['smoothing_eqn'](self, self.order, input_tuple)
        #p = self.lm_params.lm_eqn(self, self.order, input_tuple) #regular probability
        if p == 0.0:
            raise ZeroDivisionError
        if p is None:
            print('Something has a zzero-probability, use smoothing...')
            raise ValueError
        #print("p",p)
        if format == 1:
            return p
        elif format == 2:
            return math.log2(p)
        elif format == 10:
            return math.log10(p)

    def give_sentence_prob(self, sentence_list, format=1):
        """Calculate the probability of a sentence (using preset order and eqns, either log or linear depending on format."""
        #print('sentence ngrams', self.duplicate_to_ngrams(sentence_list, self.order))

        if format == 1:
            product = 1

            for tup in self.duplicate_to_ngrams(sentence_list, self.order):
                product = product * self.give_gram_prob(tup, format=format)
            return product
        elif format == 2 or format ==10: # format == 2 or 10
            return sum(self.give_gram_prob(tup, format=format) for tup in self.duplicate_to_ngrams(sentence_list, self.order))
        else:
            raise ValueError

    def give_perplexity(self, sentence_list):
        """Calculate the perplexity of a sentence."""
        print('sentence and len', sentence_list, len(sentence_list))
        print("sentence log2 prob is ", self.give_sentence_prob(sentence_list, format=2))
        entropy = -1/len(sentence_list) * self.give_sentence_prob(sentence_list, format=2)
        print('entropy is', entropy)
        return math.pow(2, entropy)


    def increment_counts(self, input):
        """Update LM counts based on input, which is either a sentence_list or another LM"""
        n = self.order

        if type(input) == list:
            range_to_build = (range(n, 0, -1) if self.lm_params.dict['save_lower_orders'] else range(n, n-2, -1))
            for order in range_to_build:
                for tup in self.duplicate_to_ngrams(input, n):
                    self.ngrams[order][tup] +=1 #Type is counter so default value 0 assumed

                    if order == self.order: # one token per new word
                        self.totalwords +=1

        elif type(input) == LanguageModel:
            #print('before', self.ngrams)
            for i in self.ngrams.keys():
                self.ngrams[i].update(input.ngrams[i])
            #print('after', self.ngrams)
        # TODO: Below are probably time-wasting
        self.vocab_size = self.calc_vocab_size()
        self.count_of_counts = self.build_count_of_counts()

    def decrement_counts(self, sentence_list):
        n = self.order
        self.range_to_build = (range(n, 0, -1) if self.lm_params.dict['save_lower_orders'] else range(n, n-2, -1))
        for order in self.range_to_build:
            for tup in self.duplicate_to_ngrams(sentence_list, n):
                self.ngrams[order][tup] -=1 #Type is counter so default value 0 assumed

                if order == self.order: # one token per new word
                    self.totalwords -=1


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
    #sentences = [['This','is','an','example','sentence'],["this","is","an","example", "blackbird"]]
    rawtext = "This is an example sentence\nthis is an example blackbird"
    rawtext = "I am Sam\nSam I am\nI do not like green eggs and ham"
    sentences = [x.split(' ')for x in rawtext.split('\n')]
    print('sents', sentences)
    # build the LM
    mylm = LanguageModel(sentences, 5, ('add-one', 'none', 'none'), holdbuild=False, bydoc=True, givedf=True)
    #args are: self, documents, n, lm, smoothing, discparams, holdbuild=False, bydoc=False, give_df=False):

    # inspect the global language model
    print("General count dictionary:", mylm.ngrams) # note that lower orders are saved based on LM combination
    print('count_of_counts', mylm.build_count_of_counts())
    print("Docfreq", mylm.docfreq)
    print("Vocabsize:", mylm.calc_vocab_size())
    print("hidden counts", mylm.deleted_symbols)

    print(mylm.give_gram_prob(("<s>", "I")))
    print(mylm.give_gram_prob(("Sam", "</s>")))
    print(mylm.give_gram_prob(("<s>","Sam", )))
    print(mylm.give_gram_prob(("am","Sam", )))
    print(mylm.give_gram_prob(("I","am", )))
    print(mylm.give_gram_prob(("I", "do")))



    # # a = mylm.give_gram_prob(["an", "example", "sentence"])
    # # b = mylm.give_perplexity(["is", "an", "foo"])
    # c = mylm.give_perplexity(["this","is","an","example","blackbird",])
    # # c2 = mylm.give_sentence_prob(["this","is","an","example","blackbird",], format=1)
    # d = mylm.give_perplexity(["This","is","an","example","blackbird",])
    # print("Seen sentence", c)
    # e = mylm.give_perplexity("This is an example sentence".split(' '))
    # print("This is example sentence", e)
    # print('perplexity', d)


    mylm.calculate_histories_continuations()
    print(mylm.continuation_counts,)
    print(mylm.histories_counts)




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


        #
        # def make_tup(x,m):
        #     return tuple(sentencelist[x : x+m])
        # def check_not_end(x): #works
        #     if x.count(LanguageModel.startsymbol) == len(x) or x.count(LanguageModel.endsymbol) == len(x):
        #         return False #give false if all sss
        #     else:
        #         return True #give true if its not sss
        #
        # if delete_sss:
        #     #print(range(len(sentencelist[n:])+1))
        #     a = [tuple(sentencelist[x : x+n]) for x in range(len(sentencelist[n:])+1)]
        #     return [y for y in a if check_not_end(y)==True]
        # else:
        #     return