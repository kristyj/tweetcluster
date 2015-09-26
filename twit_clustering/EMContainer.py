__author__ = 'Kristy'


#methods for em clustering on a clustercontainer object
import LanguageModel
from collections import deque
from collections import defaultdict
import numpy as np

'''Classes that handle building topic-based language models from documents using Expectation Maximisation.
ThetaParams holds the information about the priors of ???'''


# TODO: Replace ThetaParams with a Clustercontainer object
class ThetaParams(object):
    """Holder for the EM mixture clusters, recording prior and posteriors and LMs for each topic.
    LMs are accessible as clusterobject.clusterlms[i]
    self.topics is a list, for which every item is an EMTopicClass. These contain details about what's in the topics."""
    def __init__(self, clusterobject, lm_order, style = 'iyer-ostendorf', iters=5):  #

        #information from input
        self.style = style
        self.maxiters = iters
        self.m = clusterobject.m

        #initialise value placeholders
        self.normaliser = 1
        self.doc_associations = [[1/self.m] * self.m] # TODO: Hold in clusterobject.doc_cluster_asm

        # TODO: THese are also in Clustercontainer
        self.current_iter = 0
        self.iter_changes = [100]
        #self.do_em()


# TODO: Use clusterobject to make self.alltweets, self.totaldocs
        # #read tweets into object, to iterate over in expectation
        # self.alltweets = []
        # for c in clusterobject.clusters:
        #     self.alltweets += [t for t in c] #break into list of text already
        # self.totaldocs = len(self.alltweets)
        # print(self.totaldocs, "tweets put in a list ") ##

# TODO: Use ClusterParams to set how the topics should be initialised
        #style is passed to EMTopicClass and made into a language model of this style
        if style == 'iyer-ostendorf': #initialise probabilities on split corpus
            #initialise the topic objects from input
            self.topics = [EMTopicClass(self.m, [tweet.wordlist for tweet in clusterobject.clusters[x]], lm_order, self.totaldocs, self.style) for x in range(self.m)]
        elif style=='gildea-hoffman':
            self.topics = [EMTopicClass(self.m, [t.giveWords() for t in self.alltweets], lm_order, self.totaldocs, self.style) for x in range(self.m)]
        else:
            print("no EM style was chosen")
            exit()


    def do_em(self): #
        """initialise the expectation-maximisation procedure"""
        print("Now initialising EM on {} language models".format(self.m))
        while self.maxiters > self.current_iter and ThetaParams.measure_change(self.iter_changes):
            # while manually set iterations remain and the model changes significantly
            # #todo: check the definition of the stopping point by self.iter_changes
            self.current_iter += 1
            print("Now performing expectaton\nThis may take some time as all documents are re-read and evaluated over each topic LM.")
            self.expectation()
            print("Now performing maximisation, this re-evaluates every n-gram, therefore is slow.")
            self.maximisation()
            print("The model changed as such:", self.iter_changes)

    def __str__(self):
        return "Expectation maximisation holder, the mixes are {}".format(self.topics)

    @staticmethod
    def measure_change(somelist):
        '''For the m topics, look at the list of how much changed in the last iteration. Return True if there was a perceptible change, False if not.'''
        # TODO: Research a relevant measure
        return True if sum(somelist) > 0.01 else False

    # def give_topic_lms(self):
    #     return [topic.lm.probs for topic in self.topics]

    def expectation(self):
        print("Now performing {}th iteration, expectation step".format(self.current_iter))
        for tweet in self.alltweets:
            #theta.topic.posteriors contains zij scores # TODO: Is this w|t?
            self.calc_sentence_topic_iteration_prob(tweet.giveWords())

    def maximisation(self):
        """adjust word to topic association (w|t) based on posteriors,
        read through corpus again and update lm counts of each gram"""
        print("Now performing {}th iteration, maximisation step".format(self.current_iter))
        #clear the counts
        for topic in self.topics:
            #topic.lm.grams = {} grams retains the original counts
            topic.lm.interims = defaultdict(dict)
            topic.lm.probs = defaultdict(dict)
            topic.temp_prior = sum(topic.posteriors)

        #make a new weighted count dictionary [interims] for each bigram including the sentence weight (zij) in the topic
        print("Recounting n-grams to include weight from expectation step.")
        for tweet in self.alltweets:
            sentencelist = [LanguageModel.startsymbol]+ tweet.giveWords() + [LanguageModel.endsymbol]
            for topic in self.topics: #for each topic
                current_zij = topic.posteriors.popleft()
                for i in range(1, topic.lm.order + 1): #for each order (unigram+)
                    #make each gram
                    order_grams = LanguageModel.duplicate_to_n_grams(i, sentencelist, delete_sss=False)
                    #update the weighted counts
                    for gram in order_grams:
                        topic.lm.interims[i][gram] = topic.lm.interims[i].get(gram, 0) + current_zij #multiply by adding zij whenever encountered

        #for each topic, adjust all the probs dictionaries
        temp_total_zij = sum([topic.temp_prior for topic in self.topics])
        self.reset_posteriors()
        self.iter_changes = []

        print("Recalculating n-gram probabilities based on the new weighted counts.")
        for topic in self.topics:
            #update priors
            print('T', end='')
            self.iter_changes.append(topic.temp_prior - topic.prior) #record if it changed
            topic.prior = topic.temp_prior / temp_total_zij
            #TODO #no idea what to do for unigrams, currently don't mess with their initial probability, just normalise by the zij's collected for each unigram
            topic.lm.probs[1] = LanguageModel.counts_to_smoothed_prob(topic.lm.interims[1], sum(list(topic.lm.interims[1].values())), smoothing='nul')

            for i in range(2, topic.lm.order +1): #begin with bigrams
                topic.lm.probs[i] = {} #empty the dictionary
                for words, weightedcount in topic.lm.interims[i].items():
                    #These use the terminology of Iyer and Ostendorf pg.2
                    bqzij = topic.lm.all_other_last_words(words, weighted_count=True, include_words=False)
                    bq = topic.lm.all_other_last_words(words, weighted_count=False, include_words=False)
                    weighted_count_all_endings = sum(bqzij)
                    if weighted_count_all_endings == 0:
                        print("weighted_count_all_endings was 0") #TODO: THis means things are not the same
                        exit()

                    ml = weightedcount / weighted_count_all_endings

                    inside_fraction = sum([x / y for x, y in zip(bqzij, bq)]) #sum(q of (zij * count nbq)/(count nbq)
                    bo = ( inside_fraction ) / (weighted_count_all_endings + inside_fraction )
                    if bo >1 or ml >1:
                        print("Warning, the backoff weight exceeds 1! orthe maximum likelihood value is >1" , bo, ml, )
                        print("word", words)
                        print("weightedcount numerator", weightedcount)
                        print("Weightedcount denominator", weighted_count_all_endings)
                        exit()
                    topic.lm.probs[i][words]= (1- bo) * ml + bo * topic.lm.probs[i-1][words[1:]]
            print("checking the progression back to probability dictionary", sum(list(topic.lm.probs[1].values())))


    def __str__(self):
        return """{} mixes maintained with:\n priors: {}\n
        posteriors beginning: {}\nEach has a language model initialised on an initial cluster"""\
        .format(self.m, str([topic.posteriors[:3] for topic in self.topics]), )

    def calc_sentence_topic_iteration_prob(self, sentencelist):
        self.normaliser = 0
        #calculate numerators
        sentenceprobs =  [(topic.lm.give_sentence_probability(sentencelist, log=True)**10) for topic in self.topics]
        priors = [topic.prior for topic in self.topics]
        numerators = [sentenceprobs[x] * priors[x] for x in range(self.m)]
        #calclate the denominator
        normalizer = sum(numerators)
        zij = [numerators[x]/normalizer for x in range(self.m)]
        #extend the posteriors deque for each topic
        for ti in range(len(self.topics)):
            self.topics[ti].posteriors.extend([zij[ti]])

    def reset_posteriors(self):
        for topic in self.topics:
            topic.posteriors = deque()

    def give_as_clusters(self):
        from lib.ClusterContainer import ClusterContainer
        cluster_obj = ClusterContainer
        cluster_obj.clusters = [[] for x in range(self.m)]
        if len(self.doc_associations) < 3:
            print("There is too little info in self.doc_associations")
            exit()
        for document_number, doc_scores in enumerate(self.doc_associations):
            max_topic, max_score = max(enumerate(doc_scores), key=lambda x: x[1])
            cluster_obj.clusters[max_topic].append(self.alltweets[document_number])
        return cluster_obj


    def print_strongest_doc_associations(self):
        for topic_index, topic in enumerate(self.topics):
            print("Now printing topic {}".format(topic_index))
            for doc_index, document_scores in enumerate(self.doc_associations):
                if max(enumerate(document_scores), key = lambda x:x[1])[0]==topic_index:
                    print(self.alltweets[doc_index])
            print('topic break *******************************************************')

class EMTopicClass(object):
    def __init__(self, totalclasses, tweetlist_from_cluster, lm_order, totaldocs, style):
        '''Initiate the parameters for each topic in the EM mix - posteriors, priors, also LM information per topic'''
        #posterior is a list of the posterior for each tweet in order (essentially a list of zij for the same j)
        self.posteriors  = deque() #updated for each iteration

        if style == 'iyer-ostendorf':
        #    self.prior = 1/totaldocs #for Iyer and Ostendorf, reflecting that some topic models start with larger prior
            self.lm = LanguageModel(tweetlist_from_cluster, lm_order, smoothing='witten-bell', by_document=False)
        elif style=='gildea-hoffman':
        #    self.prior = 1/totalclasses #set at start, update at each iteration
            self.lm = LanguageModel(tweetlist_from_cluster, lm_order, smoothing='add-one', by_document=False)
        else:
            print("Define a valid style for the language model created for the EM mix")
            exit()



        #self.temp_sent_prob = float #TODO: Deal with this overflow???


class GHThetaParams(ThetaParams):
    '''EM class with functions specific to the methods used by Gildea and Hoffman.
    This means the expectation and maximisation override the Iyer Ostendorf default'''
    def __init__(self, *args, **kwargs):
        super(self.__class__,self).__init__(*args, **kwargs)
        #make a language model over all the documents recording wordcounts
        self.universal_lm = LanguageModel([t.giveWords() for t in self.alltweets], 1, smoothing='nul', by_document=True)

        # #initialise really random topic/document associations
        # self.doc_associations = []
        # for x in range(self.totaldocs):
        #     self.doc_associations.append(GHThetaParams.get_random_to_one(self.m))
        # print("This is when I first build doc_associations")
        # print(self.doc_associations)
        #self.doc_associations = [[]for x in range(self.totaldocs)]
        #Retrain self.topics

        for topic in self.topics:
            topic.posteriors = [{word: 0 for word in document} for document in self.universal_lm.tdm]#TODO initialise some list with the universal_lm_dimensions
            print(topic.posteriors[0])
        #self.do_em() #includes expectation, maximisation etc.

    # @staticmethod
    # def get_random_to_one(n):
    #     randomlist = [random.random() for a in range(n)]
    #     randomsum = sum(randomlist)
    #     #print([r/randomsum for r in randomlist])
    #     return [r/randomsum for r in randomlist]


    def expectation(self):
        #calculate P(t|w,d), ie per topic, and per topic it is per document[word]
        #initialise the normalizer as 0 for each word for each document
        ############denominator = [{(word, doc_index): 0 for word in document} for doc_index, document in enumerate(self.universal_lm.tdm)]

        for doc_index, document in enumerate(self.alltweets):
            for word in self.universal_lm.tdm[doc_index]:
                numerators = []
                for topic_index, topic in enumerate(self.topics):
                    numerators.append( topic.lm.probs[1][word] * float(self.doc_associations[doc_index][topic_index]))
                denominator = sum(numerators)

                for topic_index, topic in enumerate(self.topics):
                    new_value = float(numerators[topic_index]/denominator)
                    topic.posteriors[doc_index][word] = new_value



    def maximisation(self):
        #update P(w|t) from Gildea Hoffman m-step (1)
        for topic_index, topic in enumerate(self.topics):
            denominator = 0
            #make numerator; save denominator:
            for word in topic.lm.grams[1]: #each topic model is initialised over all documents, that could be made more efficient
                #print([self.universal_lm.tdm[doc_index].get(word, 0) * topic.posteriors[doc_index].get(word, 0) for doc_index in range(self.totaldocs)])
                try:
                    summing = 0
                    for doc_index in range(self.totaldocs):
                        count_of_word = float(self.universal_lm.tdm[doc_index].get(word, 0))
                        e_step_result = topic.posteriors[doc_index].get(word, 0) #this is where it lies....
                        summing +=  count_of_word * e_step_result
                    # topic.lm.interims[word] = sum(
                    # [self.universal_lm.tdm[doc_index].get(word, 0) * float(topic.posteriors[doc_index].get(word, 0)) for doc_index in range(self.totaldocs)]
                    # ) #numerator
                    topic.lm.interims[word] = float(summing)
                    denominator += topic.lm.interims[word]
                except TypeError:
                    print(topic_index, word, "This word didn't work.")

            for word in topic.lm.grams[1]:
                topic.lm.probs[1][word] = topic.lm.interims.get(word,0)/denominator

        #update P(t|d) from Gildea Hoffman m-step (2)
        self.doc_associations = []
        print(self.doc_associations, "Should be really empty!")
        for doc_index, document in enumerate(self.universal_lm.tdm):
            denominator = 0; numerators = []
            for topic in self.topics:
                numerators.append(sum([self.universal_lm.tdm[doc_index].get(word,0) * topic.posteriors[doc_index][word] for word in document]))
            denominator = sum(numerators)
            new_topic_associations = [n/denominator for n in numerators] #per topic, for one document
            self.doc_associations.append(new_topic_associations)

        #update P(t|d)

        #reset the information needed for the expectation step?
    def log_likelihood(self):
        pass
    def n_in_topic(self, topic, word, document):
        return self.topics[topic].lm.tdm[document].get(word, 0)
