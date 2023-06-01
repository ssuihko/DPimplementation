from sklearn.ensemble._forest import ForestClassifier
from diffprivlib.accountant import BudgetAccountant
import numpy as np
from Tree import Tree_DPDT
import random
from collections import Counter, defaultdict


class DPRF_Forest(ForestClassifier):

    def __init__(self, 
                 training_nonprocessed,
                 num_trees, # Number of trees t
                 training_dataframe, 
                 training_data, # 2D list of the training data where the columns are the attributes, and the first column is the class attribute
                 # test_data, # T
                 epsilon, # epsilon, Budget, B, the total privacy budget
                 # f, # n of attributes to be split used by each dctree f
                 max_depth, # max tree depth,
                 feature_discrete,
                 accountant=None, 
                 ):


        self._trees = []

        self.epsilon = epsilon

        self.max_depth = max_depth

        self.feature_discrete = feature_discrete

        self.num_trees = num_trees

        self.accountant = BudgetAccountant.load_default(accountant)

        ''' Some initialization '''
        # Attribute set F
        attribute_values = self.get_domains(training_data)

        # Class values set
        class_values = [str(y) for y in list(set([x[len(x)-1] for x in training_data]))]
        attribute_indexes = [int(k) for k,v in attribute_values.items()]

        
        # |D| is the sum of classes in the dataset Sigma n_c, n_c in N^C_D
        # It determines the random data points that we're going to select from D
        valid_attribute_sizes = [[int(k),len(v)] for k,v in attribute_values.items()]

        # e = B / t
        epsilon_per_tree = epsilon / float(self.num_trees)

        print("NUM TREES = {} & EPSILON PER TREE = {}".format(self.num_trees, epsilon_per_tree))
        print("Epsilon used since tree is random : ", epsilon)


        ''' minimum support threshold ''' 
        valid_attribute_sizes = [[int(k),len(v)] for k,v in attribute_values.items()]
        average_domain = np.mean( [x[1] for x in valid_attribute_sizes] )

        estimated_support_min_depth = len(training_data) / (average_domain**2) # a large number
        estimated_support_max_depth = len(training_data)  / (average_domain ** (len(attribute_indexes)/2)) # max tree depth is k/2 # a small number

        min_support = epsilon # * len(class_values)  # !!! min support threshold!

        # ? 
        root_attributes = []
       

        for a in attribute_indexes:
            if a in [x[0] for x in valid_attribute_sizes]:
                root_attributes.append(a) # OR: [index, support, gini]

        self.accountant.check(self.epsilon, 0)
    

        ''' train the trees'''
        data_sizes_for_trees = [len(el) for el in np.array_split( list(range(0, len(training_data))), num_trees)] 
        index_list = list(range(0, len(training_data)))
        # print("data sizes: ", data_sizes_for_trees) # data sizes:  [304, 304, 304, 304]

        checker = []

        # for treeId = 1,2 ... t do:
        for t in range(self.num_trees):

            if not root_attributes:
                root_attributes = attribute_indexes[:]

            # randomly extract |D| samples from D w/ a bagging algo?
            root = random.choice(root_attributes)
            root_attributes.remove(root)

            A={name: [i] for i, name in enumerate(training_dataframe.columns[:-1])}


            ''' data indices for random sampling w/o remplacement '''
            # -> bagging algorithm
            # we have t as the nth tree! 
            l = random.sample( index_list , data_sizes_for_trees[t]) # this is an indice list

            index_list = [el for el in index_list if el not in l]

            checker.append(l)

            tr_data = training_data[l]

            dataset_size = len(tr_data)

            tree = Tree_DPDT(attribute_indexes, 
                            A, # original style A
                            attribute_values, # new style attribute vals
                            root, # random root attribute for SNR
                            class_values, 
                            self.feature_discrete, 
                            'Random',  # treetype
                            dataset_size, 
                            epsilon, # epsilon per tree
                            max_depth) 

            # each tree needs to be trained on a different subsample of the training data! 
            print("currently training! ")
            tree.train(training_data[:len(tr_data)//2], 1)

            self.accountant.spend(self.epsilon, 0)
           
            print("currently pruning! ")
            tree.prune( training_data[:len(tr_data)//2], tr_data[len(tr_data)//2:] )

            # FOREST = FOREST U TREE
            self._trees.append(tree)

        # print( len(checker[0]), len(checker[1]), len(checker[2]), len(checker[3]) )

        print( list(set.intersection(*map(set, checker))) )


    def get_domains(self, data):

        attr_domains = {}
        transData = np.transpose(data)
        for i in range(0,len(data[0]) - 1):
            attr_domains[str(i)] = [str(x) for x in set(transData[i])]
        return attr_domains

    #def selective_aggregation(trees):

    #    error_rates = []

    #    for i in range(1, len(trees)):
    #        for j in range(1, len(trees) + 1 - i):
    #            STM1 =  


    #    return trees


    # GET MAJORITY LABELS (Forest, T, C) equivalent from pseudocode
    def evaluate_accuracy_with_voting(self, records, class_index):

        ''' Calculate the Prediction Accuracy of the Forest. carrot '''
        actual_labels = [x[class_index] for x in records]

        predicted_labels = []

        # class_counts = defaultdict(list)

        # FOR EACH RECORD X IN DATASET DO
        for rec in records:

            h = []

            # class_value_fractions = defaultdict(list)

            # FOR EACH TREE DO
            for tree in self._trees:

                # GET PREDICTED CLASSIFICATION RESULT for a single record
                # node, leaf_not_used = tree._classify(tree._root_node, rec)

                # print(np.array([rec])) (4 trees)

                #[['58' '15' '0' '1' '0']] * the amount of trees... (now indcludes true label?)

                # print(np.array([rec]))
                
                result = tree.pred(np.array([rec]))

                print("result: ", result)

                h.append(*result)

            # majority vote

            # IndexError: index 1 is out of bounds for axis 0 with size 1
            hh = [int(*el) for el in h]

            print("votes: ")
            print(hh) # [1, 0, 0, 0]

            c = Counter(hh)
            ans = c.most_common(1)

            predicted_labels.append(ans[0][0])

        counts = Counter([x == y for x, y in zip(map(str, predicted_labels), actual_labels)])
        
        return float(counts[True]) / len(records)