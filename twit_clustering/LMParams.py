__author__ = 'Kristy'
from twit_clustering.LMEquations import *

def overwrite_dict(da, db):
    """Do not change a dictionary, but return an 'updated' version."""
    a = da.copy()
    a.update(db)
    return a


class LMParams(object):
    """Hold parameters for the LMs, by specifying what features should be collected/calculated when building a LanguageModel.
    Can return info about the processes.
    """
    # TODO: List the possible combinations as tuples, leads to dictionary about other params.
    # these are necessary option for all combinations
    base_options = {
        'lm_info': '',
        'lm_eqn': mle_eqn,
        'smoothing_eqn': no_interpolation,
        'discount_eqn': ho_only,
        'save_lower_orders': False,
        'save_count_of_counts': False,
        'count_other_histories_futures': False,
        }

    # these are combinations that can be chosen for the LM
    setup_parameters = {
        ('maximum-likelihood', 'none', 'none'):
            overwrite_dict(base_options, {'lm_info': 'General maximum likelihood model. Operates only at the maximum order.'}),

        ('add-one', 'none', 'none'):
            overwrite_dict(base_options, {'lm_info': 'MLE with add-one smoothing',
                                          'lm_eqn': add_one_eqn,
                                          'save_lower_orders': True}),

        ('add-alpha','none','none',):
            overwrite_dict(base_options, {'lm_info': 'MLE with add-alpha smoothing',
                                          'lm_eqn': add_alpha_eqn,
                                          'save_lower_orders': True}),

        ('good-turing', 'none', 'none'):
            overwrite_dict(base_options, {'lm_info': 'Basic MLE using expected count (Good-Turing smoothing) as described in Koehn.',
                                          'lm_eqn': good_turing_eqn,
                                          'save_lower_orders' : True,
                                          'save_count_of_counts': True,}),
        ('maximum-likelihood', 'linear-interpolation', 'none'):
            overwrite_dict(base_options, {'lm_info':'Interpolated LM using basic maximum likelihood estimation.',
                                          'lm_eqn': mle_eqn,
                                          'save_lower_orders':True,
                                          'save_count_of_counts': False}),

        ('maximum-likeihood', 'recursive-backoff', 'witten-bell'):
            overwrite_dict(base_options, {'lm_info':'Recursive backoff maximum likelihood estimation using I+O\'s witten bell smoothing.',
                                          'lm_eqn': mle_eqn,
                                          'smoothing_eqn': recursive_backoff,
                                          'discount_eqn': witten_bell,
                                          'save_lower_orders':True,
                                          'save_count_of_counts': False}),
        }


    def __init__(self, order, chosen_params):
        """ See LMParams.lms_available .smoothing_available for options"""
        self.order = int(order)
        print(chosen_params)
        self.dict = LMParams.setup_parameters.get(tuple(chosen_params), self.base_options)




    def __repr__(self):
        '''Print all the object attributes'''
        return str(vars(self))

    def __str__(self):
        return 'Langauge Model parameters saved, for {} model using {} smoothing'.format(self.lm_type, self.smoothing_type)

        # def give_info(self):
        #     '''Return info about the equations used'''
        #     print(self.smoothing_type['smoothing_info'])

if __name__ == '__main__':
    myLM = LMParams(2, ['maximum-likelihood', 'abc', 'abc'])
    print(myLM)





            # check the parameters
            #print('check the params', self.lm_eqn, self.smoothing_eqn, self.discount_eqn)
            # set flags about what extra information needs to be captured in addition to ngrams
            # all_discount_info = LMParams.discount_available[self.discount_type] # TODO: Update discount available dict
            # all_smoothing_info = LMParams.smoothing_available[self.smoothing_type]
            # all_lm_info = LMParams.lms_available[self.lm_type]
            # #set the other parameters from the dictionaries
            #     setattr(self, attrname, choice)
            #     print("Setting {} option.".format(attrname))
        #                 choice = input('Please choose a valid LM parameter from the list above or type "q" to exit:')
        #                 print(options)
        #                 print("You entered {}. Invalid parameter".format(choice))
        #             else:
        #                 exit()
        #             if choice == 'q' or choice =='quit':
        #     while choice not in options:
        # for choice, options, attrname in choice_options:
        #                      ['lm_type', 'smoothing_type'])
        #                      [self.lms_available.keys(), self.smoothing_available.keys()],
        # choice_options = zip([self.lm_type, self.smoothing_type],
        # # ask for user input if no equations available
        #
        # self.lm_eqn = None; self.smoothing_eqn = None; self.param_eqn = None
        # # find the most matching equations
        #
        # self.lm_type, self.smoothing_type, self.discount_type, *_ = chosen_params
        #print('setup params', self.setup_parameters)
        # grab the input about the LM type
        #set the initial parameters
        # self.alpha = 0.04
        # self.count_other_histories_futures = False
        # self.save_count_of_counts = False
        # self.save_lower_orders =  False
        # # default params, will be written over




    # linear_interpolation = {
    #     'smoothing_info': 'Basic interpolation, taking counts from one of the LMs earlier'
    #                'Override the information about saving lower orders.',
    #     'save_lower_orders': True,
    #     'lm_eqn': linear_interpolation,
    #     'parameters' : [] ,#TODO: Populate this with a default somehow
    #     'count_other_histories_futures':False,
    # }
    # recursive_interpolation ={
    #     'smoothing_info': 'Recursive interpolation, taking counts from one of the LMs earlier'
    #     'Override the information about saving lower orders.',
    #     'save_lower_orders': True,
    #     'lm_eqn': recursive_interpolation,
    #     'parameters' : [] ,#TODO: Populate this with a default somehow
    #     'count_other_histories_futures':False,
    # }
    #
    # recursive_backoff = {
    #     'smoothing_info': 'Recursive backkoff, taking counts from one of the LMs earlier'
    #     'Override the information about saving lower orders.',
    #     'save_lower_orders': True,
    #     'lm_eqn': recursive_backoff,
    #     'parameters' : [] ,#TODO: Populate this with a default somehow
    #
    #     'count_other_histories_futures':False,
    # }
    #
    # witten_bell={
    #     'smoothing_info': 'Witten Bell smoothing used to define lambda parameters for recursive interpolation',
    #     'save_lower_orders': True,
    #     'smoothing_eqn': witten_bell,
    #     'count_other_histories_futures': True
    # }
    #
    # # TODO
    # kneser_ney={}
    # modified_kneser_ney ={}
    #
    # none_dict = {
    #     'smoothing_info':'No smoothing at all.',
    #     'save_lower_orders':False,
    #     'smoothing_eqn': no_interpolation,
    #     'count_other_histories_futures':False
    # }
    #
    # smoothing_available={
    #     'linear-interpolation': linear_interpolation,        #this moves to smoothing
    #     'recursive-interpolation': recursive_interpolation,  #this moves to smoothing
    #     'recursive-backoff': recursive_backoff,              #this moves to smoothing
    #     'none': no_interpolation,
    #     }
    #
    # discount_available = {
    #     'witten-bell':witten_bell,
    #     'kneser-ney':kneser_ney,
    #     'modified-kneser-ney':modified_kneser_ney,
    #     'good-turing': None, # TODO
    #     'none': ho_only,
    #     }

