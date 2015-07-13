__author__ = 'Kristy'
import numpy as np

class ComparisonTable(object):
    """Class handling triangular matrix that records comparison scores
    between clusters, that is, some sort of similarity score.

    Written with numpy for (or not for) ease.
    Methods:
    __int__
    __str__
    remove_row_col Y
    recalculate_when_growing Y
    find_to_merge
    initial_populate
    """

    def __init__(self, width, scorefunction):
        import numpy as np
        '''Initialise a table width x width'''
        self.width = width
        print('table will be so wide:', width)
        self.table = np.zeros((width, width))
        self.score_function = scorefunction
        print("Empty comparison table initialised")

    def __str__(self):
        '''Print some small indication of the table'''
        return "This is part of the first row"+str(self.table[0])
        #+ str([self.table[x][:max(5)] for x in range(max(5, len(self.table)))])+"..."

    def __repr__(self):
        return self.table

    def initial_populate(self, cluster_input_object):
        '''Calculate the value for every combo in the table'''
        for i in range(self.width-2):
            for j in range(i+1, self.width-1):
                self.table[i][j] = self.score_function(cluster_input_object, i, j)
        print("Table initialised as:\n", self.table)

    def find_to_merge(self):
        '''Query the scores table and give ids of the things to merge'''
        bestfound = False
        topscore = np.max(self.table, axis=(1,0))
        if topscore <= 0:
            print("Scores are all below the threshold, no more merges made")
            return None, None

        else:
            all_maxvals = np.where(self.table == topscore)
            bestranked = list(np.column_stack(all_maxvals))
            for pair in bestranked:
                if pair[0] != pair[1]:
                    smallerid, largerid = (pair[0], pair[1])
                    bestfound = True
                    return smallerid, largerid
            if bestfound == False:
                print("The find_to_merge failed because no pair to merge could be found.")
                return None, None

    def set_to_zeros(self, x, y):
        '''Put a score value of zero in all dimensions (use for any altered/merged clusters)'''
        self.table[x, :] = np.zeros(self.width)
        self.table[y, :] = np.zeros(self.width)
        self.table[:, x] = np.zeros(self.width)
        self.table[:, y] = np.zeros(self.width)


    def recalculate_when_growing(self, cluster_object, clusterid): #
        """When a cluster gains more info, reset its scores combined with everything else
        and replace with the values from scorefunction"""

        #recalculate row values from 0
        self.table[clusterid,:] = [self.score_function(cluster_object, clusterid, j)
                                   if j > clusterid and j not in cluster_object.merged_away
                                   else 0
                                   for j in range(len(self.table))]

        #recalculate column values from 0
        self.table[:, clusterid] = [self.score_function(cluster_object, i, clusterid)
                                    if i < clusterid and i not in cluster_object.merged_away
                                    else 0
                                    for i in range(len(self.table))]

        #print(self.table)

#TODO: BELOW WAS PROBABLY DEPRECATED AS THE MATRIX SIZE IS PRESERVED, BETTER TO SET TO ZERO
    # def remove_row_col(self, clusterid):
    #     '''When a cluster disappears, remove the scores for it'''
    #     try:
    #         #delete the row entries
    #         self.table = np.delete(self.table, (clusterid), axis = 0)
    #         #delete the column entries
    #         self.table = np.delete(self.table, (clusterid), axis = 1)
    #     except TypeError as err:
    #         print(err, "because you had an invalid clusterid.")