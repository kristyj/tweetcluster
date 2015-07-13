__author__ = 'Kristy'
from twit_clustering.LMEquations import *

class LMParams(object):
    """Hold parameters for the LMs, by specifying what features should be collected/calculated when building a LanguageModel.
    Can return info about the processes.
    """

    #The following dictionaries describe the parameters for each LM.
    '''
    lm_info:'',
    lm_eqn: '',
    save_lower_orders: Bool,
    save_count_of_counts:Bool,
    count_other_histories_futures:Bool,
    '''



    maximum_likelihood = {
        'lm_info': 'General maximum likelihood model. Operates only at the maximum order.',
        'lm_eqn': mle_eqn,
        'save_lower_orders' : True,
        'save_count_of_counts':False,
        'count_other_histories_futures':False,
    }
    add_one = {
        'lm_info': 'MLE with add-one smoothing',
        'lm_eqn': add_one_eqn,
        'save_lower_orders' : False,
        'save_count_of_counts':False,
        'count_other_histories_futures':False,
    }
    add_alpha ={
        'lm_info': 'MLE with add-slpha smoothing',
        'lm_eqn': add_alpha_eqn,
        'save_lower_orders' : False,
        'save_count_of_counts':False,
        'count_other_histories_futures':False,
        'alpha': 0.04,
        }
    good_turing = {
        'lm_info': 'Basic MLE using expected count (Good-Turing smoothing) as described in Koehn.',
        'lm_eqn': good_turing_eqn,
        'save_lower_orders' : True,
        'save_count_of_counts':True,
        'count_other_histories_futures':False,

    }

    lms_available = {
        'maximum-likelihood': maximum_likelihood,
        'add-one': add_one,
        'add-alpha': add_alpha,
        'good-turing': good_turing,
        'none': None
    }

    # When a class is intialised the values are read in as attributes and a object is returned



    #The following dictionaries describe the parameters for smoothing methods
    '''
    'smoothing_info':'',
    'smoothing_eqn':'',

    '''
    linear_interpolation = {
        'smoothing_info': 'Basic interpolation, taking counts from one of the LMs earlier'
                   'Override the information about saving lower orders.',
        'save_lower_orders': True,
        'lm_eqn': linear_interpolation,
        'parameters' : [] ,#TODO: Populate this with a default somehow
        'count_other_histories_futures':False,
    }
    recursive_interpolation ={
        'smoothing_info': 'Recursive interpolation, taking counts from one of the LMs earlier'
        'Override the information about saving lower orders.',
        'save_lower_orders': True,
        'lm_eqn': recursive_interpolation,
        'parameters' : [] ,#TODO: Populate this with a default somehow
        'count_other_histories_futures':False,
    }

    recursive_backoff = {
        'smoothing_info': 'Recursive backkoff, taking counts from one of the LMs earlier'
        'Override the information about saving lower orders.',
        'save_lower_orders': True,
        'lm_eqn': recursive_backoff,
        'parameters' : [] ,#TODO: Populate this with a default somehow

        'count_other_histories_futures':False,
    }

    witten_bell={
        'smoothing_info': 'Witten Bell smoothing used to define lambda parameters for recursive interpolation',
        'save_lower_orders': True,
        'smoothing_eqn': witten_bell,
        'count_other_histories_futures': True
    }

    # TODO
    kneser_ney={}
    modified_kneser_ney ={}

    none_dict = {
        'smoothing_info':'No smoothing at all.',
        'save_lower_orders':False,
        'smoothing_eqn': none_eqn,
        'count_other_histories_futures':False
    }

    smoothing_available={
        'linear-interpolation':linear_interpolation, #this moves to smoothing
        'recursive-interpolation':recursive_interpolation, #this moves to smoothing
        'recursive-backoff':recursive_backoff, #this moves to smoothing
            'witten-bell':witten_bell,
            'kneser-ney':kneser_ney,
            'modified-kneser-ney':modified_kneser_ney,
            'none':none_dict}


    def __init__(self, order, lm_type, smoothing_type):
        '''See LMParams.lms_available .smoothing_available for options'''
        #set the initial parameters
        self.order = int(order)
        self.lm_type =lm_type; self.smoothing_type = smoothing_type
        self.lm_eqn = None; self.smoothing_eqn = None

        #check the parameters
        choice_options = zip([self.lm_type, self.smoothing_type],
                             [self.lms_available.keys(), self.smoothing_available.keys()],
                             ['lm_type', 'smoothing_type'])
        for choice, options, attrname in choice_options:
            while choice not in options:
                    if choice == 'q' or choice =='quit':
                        exit()
                    else:
                        print("You entered {}. Invalid parameter".format(choice))
                        print(options)
                        choice = input('Please choose a valid LM parameter from the list above or type "q" to exit:')
            print("Setting {} option.".format(attrname))
            setattr(self, attrname, choice)

        # check the parameters
        print(self.lm_type, self.smoothing_type)

        #set the other parameters from the dictionaries
        all_lm_info = LMParams.lms_available[self.lm_type]
        all_smoothing_info = LMParams.smoothing_available[self.smoothing_type]

        for xkey, xval in all_lm_info.items():
            setattr(self, xkey, xval)

        for xkey, xval in all_smoothing_info.items():
            setattr(self, xkey, xval)

        # self.info = [self.lm_info[self.lm_type], self.smoothing_info[self.smoothing_type]]
        # self.lm_eqn =
        #
        # self.lm_eqn = self.lm_eqn_dict[self.lm_type]
        # self.smoothing_eqn = self.smoothing_eqn_dict[self.smoothing_type]
        #
        # self.save_lower_orders=True
        # self.save_count_of_counts = True
        # self.count_other_histories_futures = True



    def __repr__(self):
        '''Print all the object attributes'''
        return str(vars(self))
    def __str__(self):
        return 'Langauge Model parameters saved, for {} model using {} smoothing'.format(self.lm_type, self.smoothing_type)
    def give_info(self):
        '''Return info about the equations used'''

if __name__=='__main__':
    myLM = LMParams(2,'maximum-likelihood','abc')
    print(myLM)




