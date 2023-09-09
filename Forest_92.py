from sklearn.ensemble._forest import ForestClassifier
from diffprivlib.accountant import BudgetAccountant
import numpy as np
import random
from Node_4 import Node
from collections import Counter, defaultdict
from sklearn.metrics import mean_squared_error


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

        # Also training_data = D

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
                            'Median', # 'Median', Random' # treetype
                            dataset_size, 
                            epsilon, # epsilon per tree
                            max_depth, 
                            training_data[:len(tr_data)//2],
                            tr_data[len(tr_data)//2:]) 

            # each tree needs to be trained on a different subsample of the training data! 
            print("currently training! ")
            tree.train(  training_data[:len(tr_data)//2], 1  )

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

                #[['58' '15' '0' '1' '0']] * the amount of trees... (now indcludes true label?)

                # print(np.array([rec]))
                
                result = tree.pred(np.array([rec])) # returns the class

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





class Tree_DPDT():

    '''
    The main class of decision tree.
    '''
    def __init__(self, A_ind, A, attribute_values, root, class_values, feature_discrete, treetype, dataset_size, epsilon_per_tree, max_depth, training_data, test_data):
        '''
        attribute_values : attribute_values
        A : orignal
        feature_discrete: a dict with its each key-value pair being (feature_name: True/False),
            where True means the feature is discrete and False means the feature is 
            continuous. 
        type: ID3/C4.5/CART
        pruning: pre/post
        '''

        self.A=A
        self.attribute_values = attribute_values
        self.A_ind=A_ind
        self.feature_discrete= feature_discrete
        self.treeType=treetype
        self.leaf_count=0
        self.tmp_classification=''
        self.class_values=class_values
        self._root_node = root
        self.tree=None
        self.dataset_size=dataset_size
        self.epsilon = epsilon_per_tree # B
        self.dm = max_depth
        self.training_data = training_data
        self.test_data = test_data
        self.w = None
        self.num_classes = len(class_values)

        self.eu = epsilon_per_tree / sum(2/(self.dm - i) for i in range(0, (self.dm - 2))) + (2 / (self.dm - (self.dm-1) + 2 )) + 1
        self.ei = self.eu
        self.ei1 = 0 
        self.ei2 = self.ei
        self.current_node = None

        self.estimated_support = dataset_size / len(self.attribute_values[str( root )])

    def randOrd(self, list_of_class):
       
        count={}
        for key in list_of_class:
            count[key]=count.get(key, 0)+1

        frequency=np.array(tuple(count.values()))/len(list_of_class)
        return -1*np.vdot(frequency, np.log2(frequency))

    def Information_Gain(self, list_of_class, grouped_list_of_class):
        '''
        Compute the Information Gain.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        grouped_list_of_class: the list of class grouped by the values of 
            a certain attribute, e.g. [('duck'), ('duck', 'dolphin')].
        The Information_Gain for this example is 0.2516.
        '''
        sec2=np.sum([len(item)*self.Entropy(item) for item in grouped_list_of_class])/len(list_of_class)
        return self.Entropy(list_of_class)-sec2


    def Information_Ratio(self, list_of_class, grouped_list_of_class):
        '''
        Compute the Information Ratio.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        grouped_list_of_class: the list of class grouped by the values of 
            a certain attribute, e.g. [('duck'), ('duck', 'dolphin')].
        The Information_Ratio for this example is 0.2740.
        '''
        tmp=np.array( [len(item)/len(list_of_class) for item in grouped_list_of_class] )
        
        # Here we assume instrinsic_value is SplitInformation! 
        intrinsic_value=-1*np.vdot(tmp, np.log2(tmp))

        return self.Information_Gain(list_of_class, grouped_list_of_class) / intrinsic_value , intrinsic_value

    def exponential_method(self, options, sensitivity, epsilon, scores):

        sc_values = list(scores.values())
        sc_values2 = np.array([  sc_values  ], dtype=np.float128)

        # Calculate the probability for each element, based on its score
        probabilities = [np.exp(-(epsilon/2) * score / (2 * sensitivity)) for score in sc_values2[0]]

        # Normalize the probabilties so they sum to 1
        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

        print("prob and options lenghts: ", len(probabilities), "  ",  len(options))  # 64    71 ... 

        # Choose an element from options based on the probabilities
        return np.random.choice(options, 1, p=probabilities.astype('float'))[0]

    def exponential_method_last(self, options, sensitivity, epsilon, scores, B, N):

        sc_values = list(scores.values())
        sc_values2 = np.array([  sc_values  ], dtype=np.float128)

        # Calculate the probability for each element, based on its score
        probabilities = [np.exp(- ( (epsilon * N) / (8*B**2) ) * score / (2 * sensitivity)) for score in sc_values2[0]]

        # Normalize the probabilties so they sum to 1
        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

        print("prob and options lenghts new: ", len(probabilities), "  ",  len(options))

        return np.random.choice(options, 1, p=probabilities.astype('float'))[0]

    def MSE_calculator(self, D, A, vals, attr):  # attr: 'Age'

        ### create temporary version of tree,  where we would've split with the value and column in vals

        # save OG tree
        saved_tree = self.tree
        print("saved tree and self.tree: ", saved_tree, self.tree)

        training_data = self.training_data
        test_data = self.test_data

        split_val = vals[attr] # 37.5
        # true_vals = test_data[:, -1]

        class_index = len(training_data[0]) - 1
        actual_labels = [x[class_index] for x in test_data]
        hh_actual_labels = [int(*el) for el in actual_labels]
    
        #print("A in MSE: ")
        #print(A) # 'MaritalStatus': [3, array(['0', '1', '2', '3', '4', '5', '6']

        #print(f"unique in splitting attribute {attr} training data", np.unique(training_data[:, A[attr][0]]))
        #print(f"unique in splitting attribute {attr} in D", np.unique(D[:, A[attr][0]]))
        #print("____________")

        # temporary A that corresponds to the data in D 
        #A_copy = A.copy()
        #if self.feature_discrete[attr] is True:
        #    A_copy[attr] = [ A[attr][0], np.unique(D[:, A[attr][0]]) ]

        #print("new A copy in MSE")
        #print(A_copy)

        #print("build temp tree...")
        self.tree = self.temp_tree(training_data, A, attr, split_val) # returns a node

        #print("building temp tree worked! ", self.tree, self.tree.map)

        #print("training temp tree...")
        X = training_data[:, 0:training_data.shape[1]-1] # datarows
        X = np.array(X)

        y = training_data[:, -1] # targets
        y = [[el] for el in y]
        y = np.array(y)
        
        #print("MSE calc node loop")
        for i in range(len(X)):
            node = self.tree.classify(X[i], A )
            node.increment_class_count(y[i].item())
            #print(i, node, node._class_counts)

        self.tree.set_noisy_label(self.epsilon, self.class_values)

        #print("tree trained!")

        h = []

        for rec in test_data:
        # result = self.tree.pred(np.array([rec]))  # AttributeError: 'Node' object has no attribute 'pred'
            result = self.predict_temp( np.array([rec]), A, self.tree ) # array([['N']], dtype='<U1')
            h.append(result)

        print("new h, predictions succesful! ")

        #print(h)
    
        hh = [int(*el) for el in h]

        sc = mean_squared_error(hh, hh_actual_labels)

        print(f"sc for {attr} was: ",sc)

        self.tree = saved_tree

        return sc

    def randomOrder(self, D, A):

        '''
        Return the order by Random choise
        For the definition of D and A, see the remark in method 'fit'.
        '''

        tmp_value_dict=dict()
        med_val_dict=dict()

        #Gender
        for attr, info in A.items():

            # all the attributes possible values
            possibleVal = np.unique(D[:, info[0]])  #info[0] is the index to the array column where possible values of attribute are in D
            # this should not affect sensitivity...

            # if the continuous attribute have only one possible value, then choosing it won't improve the model, so we abandon it.
            if len(possibleVal)==1:
                continue

            if self.feature_discrete[attr] is True:

                # discrete
                if len(info) < 2:
                    A[attr].append(possibleVal)  # wonder what this means... 

                # A:  {'Gender': [3, array(['0', '1'], dtype='<U21')], 'Age': [0, 25.5], 'Education': [1, array(['0', '1', '10', '11', '12', '14', '15', '2', '4', '5', '6', '7', '8', '9'], dtype='<U21')]}

                # random value between 0 and 1
                IC_value = random.uniform(0, 1) # assign random 'values' to the attributes

                tmp_value_dict[attr] = IC_value
                
            else:

                # continuous

                split_points=(possibleVal[: -1].astype(np.float32)+possibleVal[1:].astype(np.float32))/2
                maxMetric=-1

                for point in split_points:
                    
                  
                    if len(split_points) < 2:
                        cvtrue = split_points[0]

                    else:
                        cvl = min(split_points)
                        cvm = max(split_points)

                        cvtrue = random.uniform(cvl, cvm)

                    IC_tmp = random.uniform(0, 1)

                    if IC_tmp > maxMetric:
                        maxMetric = IC_tmp
                        threshold = point

                threshold2 = cvtrue

                # set the threshold
                if len(info)<2:
                    A[attr].append(threshold2)
                   
                else:
                    A[attr][1] = threshold2

                # split points:  [30.5 32.5 37.5 44.5 48.  50.  52.  53.5 54.5 55.5 57.  61.  64.5 68.5]
                if len(split_points) < 2:
                    cv = split_points[0]

                else:
                    cv = random.uniform(0, 1)


                tmp_value_dict[attr] = cv

        
        attr_list=list(tmp_value_dict.keys())

        attr_list.sort(key=lambda x: tmp_value_dict[x])

        return attr_list

    def medianSplitOrder(self, D, A, epsilon):

        tmp_value_dict=dict()

        for attr, info in A.items():

            # all the attributes possible values
            possibleVal = np.unique(D[:, info[0]])  #info[0] is the index to the array column where possible values of attribute are in D
            # this should not affect sensitivity...

            # if the continuous attribute have only one possible value, then 
            # choosing it won't improve the model, so we abandon it.
            if len(possibleVal)==1:
                continue

            # DISCRETE SECTION - CATEGORICAL VARIABLES - MUSHROOM
            # ONE ATTRIBUTE AT THE TIME
            if self.feature_discrete[attr] is True:

                if len(info) < 2:
                    A[attr].append(possibleVal)

                Attr_ind = A[attr][0]
                
                # A:  {'Gender': [3, array(['0', '1'], dtype='<U21')], 'Age': [0, 25.5],  'Education': [1, array(['0', '1', '10', '11', '12', '14', '15', '2', '4', '5', '6', '7',
                # '8', '9'], dtype='<U21')]}

                # CALCULATE BEST SPLIT FOR EACH CATEGORICAL VARIABLE
                # THIS IS : q(C) = ||C| − |Xi \ C||
                unique, counts = np.unique( D[:, Attr_ind], return_counts=True)
                q2 = dict(zip(unique, counts))
                sc1 = {el: q2[el] - (sum(q2.values()) - q2[el]) for el in q2.keys()}
                # {'b': -623, 'c': -703, 'f': -71, 'k': -691, 's': -697, 'x': -35}
                
                opt = list(sc1.keys())

                IC_value = self.exponential_method( opt, 1, epsilon, sc1 )

                # random value between 0 and 1
                # IC_value = random.uniform(0, 1) # assign random 'values' to the attributes
                # in here the scores with exponential mechanism will be calculated... 

                # split value decided!
                tmp_value_dict[attr] = IC_value

            # CONTINUOUS SECTION - NUMERICAL VARIABLES
            else:

                # find all possible splitpoints
                # not all unique...
                split_points=( possibleVal[: -1].astype(np.float32) + possibleVal[1:].astype(np.float32))/2

                al = min(possibleVal)
                au = max(possibleVal)

                # q(r) = ||X_{i,a} ∩ [a_L, r)|−|X_{i,a} ∩ [r, a_U ]||,

                #print('continuous attr: ', attr)
                #print('attr info: ', info)
                #print("age minmax: ", al, au)
                #print('Age split_points: ')
                #print(split_points)

                ######
                an = {}

                ll = D[:, info[0]].astype(int)

                for el in np.unique(split_points):

                    a = [i for i in ll if i < el]
                    b = [i for i in ll if i >= el]
                    s = abs(len(a) - len(b))
                    an[el] = s

                ######
                # scores and options lenghts:  split_points: 64    an: 71

                print("cont scores: ", an)

                IC_value = self.exponential_method( np.unique(split_points), 1, epsilon, an )

                print('attr & ic_val: ', attr, " : ", IC_value)

                maxMetric = -1

                # Here we loop through the splits ...
                for point in split_points:
                    
                  
                    if len(split_points) < 2:
                        cvtrue = split_points[0]

                    else:
                        cvtrue = IC_value

                    if IC_value > maxMetric:

                        maxMetric = IC_value
                        threshold = point
                # Here the split loop has ended

                threshold2 = cvtrue

                # set the threshold
                if len(info) < 2:
                    A[attr].append(threshold2)
                   
                else:
                    A[attr][1] = threshold2

                if len(split_points) < 2:
                    cv = split_points[0]

                else:
                    cv = IC_value

                tmp_value_dict[attr] = cv
  
        print(tmp_value_dict) 
        # {'Age': 37.5, 'Workclass': '0',  'Education': '13',  'MaritalStatus': '1', 'Occupation': '1', 
        # 'Relationship': '2', 'Race': '3', 'HoursPerWeek': 40.5,  'Gender': '0'}

        # here all the possible attributes are listed ('gender', 'age', 'education' ... )
        attr_list = list(tmp_value_dict.keys())  
        
        ### MSE -> we're doing the calculations with the PREDICTIONS!!! (y, the target)
        ### 1rst: split D according to the value of this attribute. 
        # in here the final attr selection with exponential mechanism + MSE, will be conducted
        final_scores = {}
        
        for el in attr_list:  # 
            print("el: ", el)
            score = self.MSE_calculator(D, A, tmp_value_dict, el ) # 'Age'
            final_scores[el] = score

        # attr_list.sort(key=lambda x: tmp_value_dict[x] )
        print("READY final scores!: ", final_scores)  # {'Age': 0.2529963522668056, 'Workclass': 0.24960917144346015, 'Education': 0.23163105784262636, 'MaritalStatus': 0.25273579989577905, 'Occupation': 0.2532569046378322, 'Relationship': 0.25273579989577905, 'Race': 0.2529963522668056, 'HoursPerWeek': 0.2529963522668056, 'Gender': 0.2529963522668056}

        # this is a classification algorithm.
        Ni = len(D)
        B = 1 
    
        opt2 = list(final_scores.keys())

        final_split_value = self.exponential_method_last( opt2, 1, epsilon, final_scores, B, Ni) 

        print("chosen splitter: ", final_split_value)

        return final_split_value

    def chooseAttribute(self, D, A, eps):

        if self.treeType == 'Random':

            attr_list = self.randomOrder(D, A)

            print("chosen attr: ", attr_list[-1])

            return attr_list[-1]

        if self.treeType == 'Median':

            split = self.medianSplitOrder(D, A, eps)

            return split

    def train(self, D, depth):

        X = D[:, 0:D.shape[1]-1] # datarows
        y = D[:, -1] # targets

        X = np.array(X)

        y = [[el] for el in y]
        y = np.array(y)
        
        self.tree = self.fit(D, self.A, depth)  # fit returns a Node!

        print("self tree in train: ", self.tree)

        for i in range(len(X)):

            node = self.tree.classify(X[i], self.A)

            # 'numpy.str_' object has no attribute 'update_class_count'
            node.increment_class_count(y[i].item())

        self.tree.set_noisy_label(self.epsilon, self.class_values)


    def fit(self, D, A, depth=1):

        #print("TERMINATION CONDITIONS")
        #print("len(D): ", len(D), "uniques: ", len(np.unique(D[:, -1])), "len(A): ", len(A))
        #print("dunno, needs to be <= 1 to termintate: ", len(np.unique(D[:, :-1], axis=0)))

        #print("MAX DEPTH WORKS")
        #print("current node: ", self.current_node, " max depth: ", self.dm)
        #if self.current_node is not None:
        #    print("cur depth: ", self.current_node.depth)

        #TERMINATION CONDITIONS
        #len(D):  0 uniques:  0 len(A):  7
        #dunno, needs to be <= 1 to termintate:  0
        #MAX DEPTH WORKS
        #current node:  <__main__.Node object at 0x7ff51c2bc220>  max depth:  50
        #cur depth:  4

        min_support = self.epsilon

        # print(" for leaf node creation ", min_support, " must be larget than ", self.estimated_support)


        ''' termination conditions '''
        # the training set is empty
        if len(D)==0:

            node = Node(feature_name='leaf-'+str(self.leaf_count), 
                        depth=depth, 
                        isLeaf=True,
                        classification=self.tmp_classification)

            self.leaf_count+=1
            return node

        # only one type of classification is left 
        if len(np.unique(D[:, -1])) <= 1:
            node = Node(feature_name='leaf-'+str(self.leaf_count), 
                        depth=depth, 
                        isLeaf=True,
                        classification=D[0, -1])
            self.leaf_count+=1
            return node

        if len(A) == 0 or len(np.unique(D[:, :-1], axis=0)) <= 1:

            count_dict={}

            for key in D[:, -1]:

                count_dict[key]=count_dict.get(key, 0) + 1

            most_frequent=sorted(D[:, -1], key=lambda x: count_dict[x])[-1]

            node=Node(feature_name='leaf-'+str(self.leaf_count), 
                      depth=depth, 
                      isLeaf=True, 
                      classification=most_frequent)

            self.leaf_count+=1
            return node

        
        # stop building at max depth!
        if self.current_node is not None: 
            if self.dm <= self.current_node.depth:

                print("terminated at :", self.current_node.depth, " deep")

                print("len(D): ", len(D), "uniques: ", len(np.unique(D[:, -1])), "len(A): ", len(A))

                count_dict={}

                for key in D[:, -1]:
                    count_dict[key]=count_dict.get(key, 0)+1

                most_frequent=sorted(D[:, -1], key=lambda x: count_dict[x])[-1]

                node = Node(feature_name='leaf-'+str(self.leaf_count), 
                            depth=depth, 
                            isLeaf=True, 
                            classification=most_frequent)

                self.leaf_count+=1

                return node


        '''continue tree building '''
        # stop conditions did not apply -> continue building the tree

        # self.w = 2 / (self.dm - self.current_) # not completely correct
        self.budget = self.eu # (e / w)

        count_dict={}

        # the count query usese the Laplace mechanism to add noise to the class count
        for key in D[:, -1]:
            count_dict[key] = count_dict.get(key, 0)+ 1 + np.random.laplace(scale=(1/self.budget/2))

        # print("count dictionary: ")
        # print(count_dict)
        # count query : how many records have each class value? 
        # {'1': 449, '0': 162}
        # {'0': 4, '1': 5}
        
        most_frequent = sorted(D[:, -1], key=lambda x: count_dict[x])[-1]
      
        # count dictionary: 
        #{'0': 4, '1': 1}
        # most frequent : 
        # 0

        self.tmp_classification = most_frequent

        # choosing the target attribute -> should be random!
        target_attr = self.chooseAttribute(D, A, self.budget/2)  # Race, Age

        # generate nodes for each possible value of the target attribute if it's discrete
        # related information is stored in A[target_attr][1] now, 
        # since we have called chooseAttribute at least once.
        # "divide the current node to MULTIPLE CHILD NODES according to class labels
        # a new node
        

        info = A[target_attr] # info[1] should hold the split value
        # chosen attr:  Workclass
        #info
        #[1, array(['0', '1', '3', '4', '5', '6'], dtype='<U21')] -> this tree creates a child for each chosen attr value for cat. columns

        print("info")
        print(info)

        if self.feature_discrete[target_attr]:

            node = Node(feature_name=target_attr,  discrete=True,  depth=depth,  isLeaf=False)


            self.current_node = node

            for possibleVal in info[1]:

                # important, for this affects tmp_A
                # keys is just the names of the attributes ("Age", "Gender" etc.) without the target attr name.
                keys = set(A.keys()).difference({target_attr})  # all keys but the chosen column

                # connect node to its child???? -> divides D based on all the values of categorical column defined by the index info[0]
                tmp_D = D[np.argwhere( D[:, info[0]]==possibleVal), :]
                
                tmp_A = {key: A[key] for key in keys}  
                
                # this here calls  def __setitem__(self, key, value):

                #print("given as input to node[possibleVal] ")
                #print("reshaper: ")
                #print(  tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])))
                # print(tmp_A)

                #reshaper: 
                #[['37' '5' '11' '0' '0' '4' '4' '12' '0' '1']
                #['62' '2' '15' '0' '0' '4' '4' '30' '0' '1']]
                
                #tmp_A: 
                #{'Gender': [8, array(['0', '1'], dtype='<U21')], 
                # 'HoursPerWeek': [7, 48.764268089700906], 
                # ...
                # 'Age': [0, 37.56920998722627], 
                # 'Education': [2, array(['0', '1', '10', '11', '12', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U21')]}

                node[possibleVal] = self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), tmp_A, depth + 1)
               
                # added to discrete children
                # this here calls  def __getitem__():
                node.add_cat_child(possibleVal, node[possibleVal])

        
        else:
            # generate two nodes for the two classification if it's continuous
            # continuous

            threshold = info[1]  # 'HoursPerWeek': [7, 48.764268089700906], 'Age': [0, 37.56920998722627],
            # treshold_valie = np.random.uniform()

            #print("domains in self.attribute_values")
            #print(self.attribute_values)
            #print("target attr")

            # confused over how the split value gets found during the continuous attribute node split...

            # target attr:  Age treshold:  28.5
            node=Node(feature_name=target_attr,  discrete=False,  threshold=threshold,  depth=depth,  isLeaf=False)

            self.current_node = node

            tmp_D=D[np.argwhere(D[:, info[0]]<=str(threshold)), :]
            node['<='] = self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A, depth+1)

            tmp_D=D[np.argwhere(D[:, info[0]]>str(threshold)), :]
            node['>'] = self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A, depth+1)

            node.set_left_child( node['<='] )
            node.set_right_child( node['>'] )
            
        
        return node

        #########################
        #########################

    def prune(self, training_D, testing_D):
        self.post_prune(training_D, testing_D, self.A, current = self.tree)

    def post_prune(self, training_D, testing_D, A, current=None, parent=None):
    
        '''
        self.tree is required.
        This method conducts the post-pruning to enhance the model performance.
        To make sure this method will work, set 
        >> current=self.tree
        when you call it.
        '''

        # print( self.tree.map.items() )
        # dict_items([
        # ('b', <__main__.Node object at 0x7ff51e8850d0>), 
        # ('c', <__main__.Node object at 0x7ff51e885610>), 
        # ('f', <__main__.Node object at 0x7ff51e885640>), 
        # ('k', <__main__.Node object at 0x7ff51e8a4520>), 
        # ('s', <__main__.Node object at 0x7ff51e885ee0>), 
        # ('x', <__main__.Node object at 0x7ff51e885760>)])

        self.current_accuracy = self.evaluate( testing_D, A )

        # if DB is empty
        if len(training_D)==0:
            return 

        count_dict={}
        for key in training_D[:, -1]:
            count_dict[key] = count_dict.get(key, 0)+1

        most_frequent=sorted(training_D[:, -1], key=lambda x: count_dict[x])[-1]

        #count dict: 
        #{'1': 6, '0': 6}
        #most frequent: 
        #1

        leaf_parent = True

        for key, node in current.map.items():

            if not node.isLeaf:
                leaf_parent = False

                # Recursion, DFS, Depth First Traversal
                if node.discrete:

                    tmp_D=training_D[np.argwhere(training_D[:, A[current.feature_name][0]]==key), :]

                else:
                    if key=='<=':
                        tmp_D = training_D[np.argwhere(training_D[:, A[current.feature_name][0]]<=str(node.threshold)), :]
                    
                    else:
                        tmp_D = training_D[np.argwhere(training_D[:, A[current.feature_name][0]]>str(node.threshold)), :]
                
                # find leaf node
                self.post_prune( tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])) , testing_D, A, parent=current, current=node)
        

        print("leaf count: ", self.leaf_count)
                                                #?
        tmp_node = Node( feature_name='leaf-'+str(self.leaf_count), isLeaf = True, classification=most_frequent )
        


        if parent:

            # when current node is not the root
            for key, node in parent.map.items():
                if node == current:
                    parent.map[key] = tmp_node
                    saved_key = key
                    break

            # compare the evaluation, if it is enhanced then prune the tree.
            tmp_accuracy = self.evaluate(testing_D, A)

            if tmp_accuracy < self.current_accuracy:
                parent.map[saved_key] = current
                
            else:
                self.current_accuracy = tmp_accuracy
                self.leaf_count += 1

            return

        else:

            # when current node is the root -> parent: None, we've reached the top!
            saved_tree = self.tree
            self.tree = tmp_node

            tmp_accuracy = self.evaluate(testing_D, A)

            if tmp_accuracy < self.current_accuracy:
                self.tree = saved_tree

            else:
                self.current_accuracy = tmp_accuracy
                self.leaf_count += 1

            return


    def pred(self, D):
        return self.predict(D, self.A)
    
    def eval(self, D):
        return self.evaluate(D, self.A)


    def predict(self, D, A):
        '''
        Predict the classification for the data in D.
        For the definition of A, see method 'fit'. 
        There is one critical difference between D and that defined in 'fit':
            the last column may or may not be the labels. 
            This method works as long as the feature index in A matches the corresponding
            column in D.
            apple
        '''

        # why the loop can not be dismissed?
        row, _= D.shape # for the entire testing data 
        pred = np.empty((row, 1), dtype=str)

        # nodes for the exp mech
        pred_node = np.empty((row, 1), dtype=str)

        tmp_data={key: None for key in A.keys()}

        # print("the tree is a: ", self.tree, type(self.tree))
        # the tree is a:  <class '__main__.Node'>

        #print("A in predict")
        #print(A) # {'MaritalStatus': [3, array(['0', '2', '3', '4', '5', '6'], dtype='<U21')], 

        for i in range(len(D)): # only 1 row?
            for key, info in A.items():

                tmp_data[key] = D[i, info[0]]

            #print("tmp_data in predict")
            #print(tmp_data)   # {'MaritalStatus': '2', 'Relationship': '0', 'Workclass': '3', 'HoursPerWeek': '38', 'Gender': '1', 'Age': '28', 'Race': '4', 'Occupation': '5'}

            # but self.tree is evidently initialized as none?
            # the idea is that by calling this self.tree(tmp_data) the trained tree would have nothing extra compared to data
            pred[i] = self.tree(tmp_data) # -> calls __call__ in node

            # node = self.tree.classify(X[i], self.A)
            # datatype of x and x:  ['57' '15' '10' '1'] <class 'numpy.ndarray' 
            # datatype of fn and fn:  dict_values(['23', '15', '7', '1']) <class 'numpy.ndarray'>
            fn = np.array([tmp_data[el] for el in tmp_data])

            pred_node[i] = self.tree.classify(fn, A, diction=True).noisy_label

            # print("pred[i] in pred loop: ", pred[i], pred_node[i]) # pred[i] in pred loop:  ['1']
      
        return pred_node
    
    def evaluate(self, testing_D, A):  ## returns accuracy, not MSE
       
        true_label = testing_D[:, -1]
        pred_label = self.predict(testing_D, A)
        
        success_count=0
        for i in range(len(true_label)):
            if true_label[i]==pred_label[i]:
                success_count+=1

        return success_count/len(true_label)


    # the splitting happens here! 
    def temp_tree(self, D, A, target_attr, split_val):

        info = A[target_attr] 

        #print(f"unique in temp_tree's D for target {target_attr}") "Occupation"
        #print(np.unique(D[:, info[0]])) # ['0' '1' '10' '11' '12' '13' '2' '3' '4' '5' '6' '7' '8' '9']

        if self.current_node is not None:
            depth = self.current_node.depth + 1
        else:
            depth = 1

        # corner case - all of the same class...

        # print(f"were going to split with the value {split_val} of column {target_attr}")

        if self.feature_discrete[target_attr]:

            #print("Attribute recognized as discrete")
            #print("info: ")
            #print(info) # [3, array(['0', '1', '2', '3', '4', '5', '6'], dtype='<U21')]

            node = Node(feature_name=target_attr,  discrete=True,  depth=depth,  isLeaf=False)

            # well... I could already do the thing? 

            for possibleVal in info[1]: # for all the values in the categorical...

                keys = set(A.keys()).difference( {target_attr} )

                #if split_val == possibleVal:
                #    print("temp tree true, ", split_val, possibleVal)
                tmp_D = D[np.argwhere(D[:, info[0]] == possibleVal), :]

                tmp_A = {key: A[key] for key in keys}

                node[possibleVal] = self.fit_temp( tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), tmp_A, depth + 1)

                node.add_cat_child(possibleVal, node[possibleVal])

        else:

            threshold = info[1]
            print("correct treshold? : ", threshold)  

            node=Node(feature_name=target_attr,  discrete=False,  threshold=threshold,  depth=depth,  isLeaf=False)
            tmp_D=D[np.argwhere(D[:, info[0]]<=str(threshold)), :]

            node['<='] = self.fit_temp(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A, depth+1)

            tmp_D=D[np.argwhere(D[:, info[0]]>str(threshold)), :]

            node['>'] = self.fit_temp(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A, depth+1)

            node.set_left_child( node['<='] )
            node.set_right_child( node['>'] )

        return node

    def fit_temp(self, D, A, depth):

        ''' termination conditions '''
        count_dict={}

        for key in D[:, -1]:

            count_dict[key]=count_dict.get(key, 0)+1

        #print("In A")
        #print(A)

        #print("count_dict")
        #print(count_dict)

        #print("in D")
        #print(D)

        most_frequent = sorted(D[:, -1], key=lambda x: count_dict[x])[-1]

        node = Node(feature_name='leaf-'+str(self.leaf_count),  depth=depth,  isLeaf=True, classification=most_frequent)

        return node

    def predict_temp(self, D, A, tree):
  
        # why the loop can not be dismissed?
        row, _= D.shape # for the entire testing data 
        pred = np.empty((row, 1), dtype=str)
        pred_node = np.empty((row, 1), dtype=str)

        tmp_data={key: None for key in A.keys()} 

        for i in range(len(D)): # only 1 row?
            for key, info in A.items():

                tmp_data[key] = D[i, info[0]]

            pred[i] = tree(tmp_data) # -> calls __call__ in node

            fn = np.array([tmp_data[el] for el in tmp_data])

            # pred_node[i] = tree.classify(fn, A, diction=True).noisy_label # not all nodes fetched by this function have a ready noisy_label... why?

            #print("pred[i] in pred loop: ", pred[i], pred_node[i]) # pred[i] in pred loop:  ['1']
      
        return pred