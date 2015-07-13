__author__ = 'Kristy'



# def get_docids_from_files(filenames):
#     #read the document ids into lists
#     m = len(files)
#     ids  = [[] for x in range(m)]
#     for k, filename in enumerate(filenames):
#         current_file = open(filename, 'r', encoding='utf-8')
#         for line in current_file:
#             id, text = line.strip('\n').split('\t')
#             ids[k].append(id)
#     return ids
#
# def align_clusters_to_default(defaultclust, *otherclusts):
#     '''Take the clusters and swap the order so they best conform with the default cluster's order.'''
#     #comparing only two clusters
#     if type(otherclusts[0])!=list:
#         print('You should ideally to compare at least three clusters')
#         pass
#     #compare default and 2+ clusters
#     else:
#         for otherclust in otherclusts:
#             #find the best alignment between defaultclust and otherclust
#             for k in otherclust:
#                 #align the points as best possible.
#
# #m filenames with the id, text info
# m1_files = []
# m2_files = []
# m1 = get_docids_from_files(m1_files)
# m2 = get_docids_from_files(m2_files)
# m = len(m1_files)
# allmethods = [m1,m2]
# #etc... Better to streamline this

#enter the raw data about what is in which cluster

method1 = [[1,2,4,8],[3,7],[5,9,0]]
method2 = [[5,9,0],[1,7,8],[2,4,3]]
method3 = [[1,3,4],[9,8,7,2],[0,5]]
methodnames = ['method1name', 'method2name', 'method3name']

allmethods = [method1,method2, method3] #group all methods in one list


num_methods = len(allmethods); num_clusters = len(method1)
all_ids = [item for sublist in method1 for item in sublist]
#print(all_ids)

#create a dictionary that gives x,y coordinates
clust_doc_coord = {x: {} for x in range(num_methods)}

for methodnum, method in enumerate(allmethods):
    for clusternum, cluster in enumerate(method):
        for docid in cluster:
            clust_doc_coord[methodnum][docid] = [clusternum, methodnum]
print(clust_doc_coord)

#create a dictionary that gives the color based on the default cluster
cluster_color = ['b','r','y']
doc_color = {doc_id: cluster_color[cluster_id] for cluster_id, cluster in enumerate(method1) for doc_id in cluster}




from matplotlib import pyplot as plt
import numpy as np
m = 3


def print_changes_over_methods():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n in range(num_methods):
        plt.scatter([x for x in range(m)],[n for x in range(m)], s=[30*2**len(x) for x in allmethods[n]], color='k')
        ax.text(0, n, methodnames[n])
        if n>=1: #only do this where there is a comparison
            for doc_id in all_ids:
                oldcoords = clust_doc_coord[n-1][doc_id]
                newcoords = clust_doc_coord[n][doc_id]
                jitter = np.random.normal(0, 0.01)
                plt.plot([oldcoords[0]+jitter, newcoords[0]+jitter],[oldcoords[1], newcoords[1]],
                         color = doc_color[doc_id])
    plt.show()







exit()

xpoints = [x for x in range(m)] #build some points along the x-axis
defaulty = [0 for x in range(m)]
plt.plot(xpoints, defaulty)


plt.show()

for number, method in enumerate(allmethods):
    ypoints = [number for y in range(m)] #builds an equal number of points on y-axis
    plt.plot([],[])




