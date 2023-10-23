from sklearn.ensemble._forest import ForestClassifier
from diffprivlib.accountant import BudgetAccountant
import numpy as np
import random
# from Node_10 import Node
from collections import Counter, defaultdict
from sklearn.metrics import mean_squared_error
from collections import Counter, defaultdict
from diffprivlib.mechanisms import PermuteAndFlip
from diffprivlib.mechanisms import Exponential 
from statistics import mode
import pandas as pd
import math


class DPRF_Forest(ForestClassifier):

    def __init__(self, 
                 training_nonprocessed,
                 num_trees, # Number of trees t
                 training_dataframe, 
                 training_data, # 2D list of the training data where the columns are the attributes, and the first column is the class attribute
                 # test_data, # T
                 epsilon, # epsilon, Budget, B, the total privacy budget
                 max_depth, # max tree depth,
                 K,
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

        
        # Determines the random data points that we're going to select from D
        valid_attribute_sizes = [[int(k),len(v)] for k,v in attribute_values.items()]

        # e = B / t
        epsilon_per_tree = epsilon / float(self.num_trees)  # 

        print("Epsilon used: ", epsilon)

        valid_attribute_sizes = [[int(k),len(v)] for k,v in attribute_values.items()]

        root_attributes = []
       
        for a in attribute_indexes:
            if a in [x[0] for x in valid_attribute_sizes]:
                root_attributes.append(a)

        self.accountant.check(self.epsilon, 0)
    

        # randomize training data order
        np.random.shuffle(training_data)

        ''' train the trees'''
        data_sizes_for_trees = [len(el) for el in np.array_split( list(range(0, len(training_data))), num_trees)] 
        index_list = list(range(0, len(training_data))) # [0, 1, 2, 3, ... , 1234]

        print("datasizes: ", data_sizes_for_trees)

        # for treeId = 1,2 ... t do:
        for t in range(self.num_trees):

            if not root_attributes:
                root_attributes = attribute_indexes[:]

            root = random.choice(root_attributes)
            root_attributes.remove(root)

            A = {name: [i] for i, name in enumerate(training_dataframe.columns[:-1])} #only holds the index

            # print("A: ", A)

            ''' data indices for random sampling w/o remplacement '''
            l = random.sample( index_list , data_sizes_for_trees[t] ) # this is an indice list | sample this much indices from index_list
            index_list = [el for el in index_list if el not in l]     # remove these indices from index_list -> no repeats

            tr_data = training_data[l]
            dataset_size = len(tr_data)

            tree = Tree_DPDT(attribute_indexes, 
                            A, # original style A
                            attribute_values, # new style attribute vals
                            root,
                            class_values, 
                            self.feature_discrete, 
                            'Median', # 'Median', Random' # treetype
                            K,
                            dataset_size, 
                            epsilon, # epsilon per tree
                            max_depth, 
                            tr_data, #training_data[:len(tr_data)//2], 
                            tr_data[len(tr_data)//2:] ) 

            
            # each tree needs to be trained on a different subsample of the training data! 
            print("currently training! ")
            tree.train(  tr_data, 1  )  # training_data[:len(tr_data)//2]

            self.accountant.spend(self.epsilon, 0)
           
            # FOREST = FOREST U TREE
            self._trees.append(tree)



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

        for rec in records:

            h = []

            # FOR EACH TREE DO
            for tree in self._trees:

                result = tree.pred( np.array([rec]) ) # returns the class

                h.append(*result)

            # majority vote
       
            ht = [item for sublist in h for item in sublist]
            htt = [el for el in ht if el != 'N']
            
            hh = [float(*el) for el in htt]

            #print("votes: ")
            #print(hh) # [1, 0, 0, 0]

            c = Counter(hh)
            ans = c.most_common(1)

            predicted_labels.append(ans[0][0])

        counts = Counter([x == y for x, y in zip(map(str, predicted_labels), actual_labels)])
        
        return float(counts[True]) / len(records)





class Tree_DPDT():

    '''
    The main class of decision tree.
    '''
    def __init__(self, A_ind, A, attribute_values, root, class_values, feature_discrete, treetype, K, dataset_size, epsilon_per_tree, max_depth, training_data, test_data):
        '''
        attribute_values : attribute_values
        A : orignal
        feature_discrete: a dict with its each key-value pair being (feature_name: True/False),
            where True means the feature is discrete and False means the feature is 
            continuous.
        '''

        self.A=A
        self.attribute_values = attribute_values
        self.A_ind=A_ind
        self.feature_discrete= feature_discrete
        self.treeType=treetype
        self.leaf_count=0
        self.tmp_classification=''
        self.tmp_classification_2=''
        self.class_values=class_values
        self._root_node = root
        self.tree=None
        self.dataset_size=dataset_size
        self.epsilon = epsilon_per_tree # B
        self.dm = max_depth
        self.epsilon_sa = (0.5 * epsilon_per_tree) / (2 * max_depth)  
        self.epsilon_l = 0.5 * epsilon_per_tree
        self.training_data = training_data
        self.test_data = test_data
        self.w = None
        self.num_classes = len(class_values)
        self.current_node = None
        self.K = K
        self.estimated_support = dataset_size / len(self.attribute_values[str( root )])

    def mse(self, data):
        data = pd.Series(data)
        res = np.mean((data - data.mean())**2)
        if np.isnan(res):
            return 1
        else:
            return res

    def mse_gain(self, left, right, current_mse):

        w = float(len(left)) / (len(left) + len(right))
        return current_mse - w * self.mse(left) - (1 - w) * self.mse(right)


    def exponential_method(self, options, sensitivity, epsilon, scores):

        sc_values = list(scores.values())

        sc3 = [-el for el in sc_values]

        mech = Exponential(epsilon=epsilon, 
                            sensitivity=sensitivity,
                            monotonic=True, 
                            utility=sc3,
                            candidates=list(options))

        res = mech.randomise()

        return res

    def exponential_method_last(self, options, sensitivity, epsilon, scores, B, N):

        sc_values = list(scores.values())

        sc3 = [-el for el in sc_values]

        mech = Exponential(epsilon=epsilon, 
                            sensitivity=4/N,
                            monotonic=True, 
                            utility=sc3,
                            candidates=list(options))

        res = mech.randomise()

        return res

    def simple_MSE_calculator(self, D, A, vals, attr):  # attr: 'Age'

        split_val = vals[attr]   
        ind = A[attr]

        training_data = D #self.training_data
        test_data = self.test_data
        split_val = vals[attr]            

        y = D[:, -1]
        y = [[el] for el in y]

        y2 = D[: ,-1]
        y2 = [float(el) for el in y2]
        y_train = np.array(y2)


        if not self.feature_discrete[attr]: # cont

           
            D1 = D[np.argwhere(D[:, ind[0]] <= str(split_val)), :]
            D2 = D[np.argwhere(D[:, ind[0]] > str(split_val)), :]
            D1 = D1.reshape((D1.shape[0], D1.shape[2]))
            D2 = D2.reshape((D2.shape[0], D2.shape[2]))

            DY1 = D1[:, -1]
            DY2 = D2[:, -1]

            DY1 = np.array([float(el) for el in DY1])
            DY2 = np.array([float(el) for el in DY2])

            m1, m2 = np.mean(DY1), np.mean(DY2)
            l1, l2 = len(DY1), len(DY2)

            r1 = np.sum([(el - m1)**2 for el in DY1])
            r2 = np.sum([(el - m2)**2 for el in DY2])

            mse_attr = r1 + r2 

        else:

            D1 = training_data[np.argwhere(training_data[:, ind[0]] == split_val), :]
            D2 = training_data[np.argwhere(training_data[:, ind[0]] != split_val), :]
            D1 = D1.reshape((D1.shape[0], D1.shape[2]))
            D2 = D2.reshape((D2.shape[0], D2.shape[2]))

            DY1 = D1[:, -1]
            DY2 = D2[:, -1]

            DY1 = np.array([float(el) for el in DY1])
            DY2 = np.array([float(el) for el in DY2])

            m1, m2 = np.mean(DY1), np.mean(DY2)
            l1, l2 = len(DY1), len(DY2)

            r1 = np.sum([(el - m1)**2 for el in DY1])
            r2 = np.sum([(el - m2)**2 for el in DY2])

            mse_attr = r1 + r2 

        return mse_attr

    def medianSplitOrder(self, D, A, epsilon):

        tmp_value_dict=dict()

        AA = A.copy()

        def attr_sub(Atr, amount):

            am = len(Atr)
            ans = min(am, amount)

            A_rand = dict(random.sample( Atr.items(), ans ))

            idr = [A_rand[a][0] for a in A_rand.keys()]
            pv = [ len( np.unique(D[:, a]) ) for a in idr ] 

            return A_rand, pv

        A_rand, check = attr_sub(A, self.K)

        if sum(check) == len(check):
          
            BB = {key: AA[key] for key in AA if key not in A_rand.keys() }
            BB2 = {key: self.A[key] for key in self.A if key not in A_rand.keys() }

            A_rand, pv = attr_sub( BB, self.K )

            if sum(pv) == len(pv):

                ms = {el: BB2[el] for el in BB2 if el not in A.keys()} 
                dict2 = {el: BB2[el] for el in ms.keys()}
                A.update(dict2)
                
                A_rand, pv = attr_sub( BB2, len(BB2) )

                print("pv: ", pv)

        for attr, info in A_rand.items():

            # all the attributes possible values
            possibleVal = np.unique(D[:, info[0]])

            if len(possibleVal)==1:
                continue

            if self.feature_discrete[attr] is True:

                if len(info) < 2:
                    A[attr].append(possibleVal)

                Attr_ind = A[attr][0]
                
                unique, counts = np.unique( D[:, Attr_ind], return_counts=True)
                q2 = dict(zip(unique, counts))
             
                sc1 = {el: abs(q2[el] - (sum(q2.values()) - q2[el])) for el in q2.keys()}
           
                opt = list(sc1.keys())

                IC_value = self.exponential_method( opt, 1, self.epsilon_sa, sc1 )

                tmp_value_dict[attr] = IC_value

            else:

                # find all possible splitpoints  not all unique...
                split_points= ( possibleVal[: -1].astype(np.float32) + possibleVal[1:].astype(np.float32))/2

                al, au = min(possibleVal), max(possibleVal)

                # q(r) = ||X_{i,a} ∩ [a_L, r)|−|X_{i,a} ∩ [r, a_U ]||,
                an = {}

                ll = D[:, info[0]].astype(float)

                sp2 = np.linspace(float(al), float(au), 12)[1:11]

                for el in np.unique(sp2):

                    a = [i for i in ll if i < el]
                    b = [i for i in ll if i >= el]
                    s = abs(len(a) - len(b)) # abs()
                    an[el] = s


                IC_value = self.exponential_method( np.unique(sp2), 1, self.epsilon_sa, an )
                
                threshold2 = IC_value

                # set the threshold
                if len(info) < 2:
                    A[attr].append(threshold2)
                   
                else:
                    A[attr][1] = threshold2

                tmp_value_dict[attr] = IC_value


        attr_list = list(tmp_value_dict.keys())  

        final_scores = {}
        
        for el in attr_list:  

            score = self.simple_MSE_calculator(D, A, tmp_value_dict, el ) 
            final_scores[el] = score

        Ni = len(D)
        B = 1 

        opt2 = list(final_scores.keys())

        final_split_value = self.exponential_method_last( opt2, 2, self.epsilon_sa, final_scores, B, Ni)
    
        return final_split_value, tmp_value_dict[ final_split_value ]

    def chooseAttribute(self, D, A, eps):

        if self.treeType == 'Median':

            split, split_value = self.medianSplitOrder(D, A, eps)

            return split, split_value

    def train(self, D, depth):

        X = D[:, 0:D.shape[1]-1] # datarows
        y = D[:, -1] # targets

        X = np.array(X)

        y = [[el] for el in y]
        y = np.array(y)
        
        self.tree = self.fit(D, self.A, depth)  # fit returns a Node!

        for i in range(len(X)):

            node = self.tree.classify(X[i], self.A)

            node.increment_class_count(y[i].item())

        # place half of the budget to creating the noisy labels
        self.tree.set_noisy_label(self.epsilon_l, self.class_values)



    def fit(self, D, A, depth=1):

        ''' termination conditions '''
        
        if len(D)==0:

            node = Node(feature_name='leaf-'+str(self.leaf_count), 
                        depth=depth, 
                        isLeaf=True,
                        classification=self.tmp_classification)

            self.leaf_count+=1

            return node

        if len(np.unique(D[:, -1])) <= 1:
            node = Node(feature_name='leaf-'+str(self.leaf_count), 
                        depth=depth, 
                        isLeaf=True,
                        classification=D[0, -1])
            self.leaf_count+=1
            return node

        if len(A) == 0 or len(np.unique(D[:, :-1], axis=0)) <= 1:  #or len(D) < 4:

            count_dict={}

            for key in D[:, -1]:

                count_dict[key]=count_dict.get(key, 0) + 1

            most_frequent=sorted(D[:, -1], key=lambda x: count_dict[x])[-1]

            utility = list( count_dict.values() ) 
            candidates = list( count_dict.keys())      

            mech = PermuteAndFlip(epsilon=self.epsilon_l, 
                            sensitivity=1,
                            monotonic=True, 
                            utility=utility,
                            candidates=candidates)

            cl = mech.randomise()

            node=Node(feature_name='leaf-'+str(self.leaf_count), 
                      depth=depth, 
                      isLeaf=True, 
                      classification=cl)

            self.leaf_count+=1
            return node

        # stop building at max depth!
        if self.current_node is not None: 
            if self.dm <= self.current_node.depth:

                count_dict={}

                for key in D[:, -1]:
                    count_dict[key]=count_dict.get(key, 0)+1

                most_frequent=sorted(D[:, -1], key=lambda x: count_dict[x])[-1]

                utility = list( count_dict.values() )
                candidates = list( count_dict.keys())      

                mech = PermuteAndFlip(epsilon=self.epsilon_l, 
                            sensitivity=1,
                            monotonic=True, 
                            utility=utility,
                            candidates=candidates)

                cl = mech.randomise()

                node = Node(feature_name='leaf-'+str(self.leaf_count), 
                            depth=depth, 
                            isLeaf=True, 
                            classification=cl )

                self.leaf_count+=1

                return node

        '''continue tree building '''
      
        # temporary label 
        count_dict={}

        # the count query usese the Laplace mechanism to add noise to the class count
        for key in D[:, -1]:
            count_dict[key] = count_dict.get(key, 0)+ 1

        utility = list( count_dict.values() ) # f_c -> frequency of the class count
        candidates = list( count_dict.keys())

        # for temporary noisy label if D_split=0
        mech = PermuteAndFlip(epsilon=self.epsilon_l, 
                            sensitivity=1,
                            monotonic=True, 
                            utility=utility,
                            candidates=candidates)

        self.tmp_classification = mech.randomise()

        # half of the epsilon goes to the splitting 
        target_attr, split_value = self.chooseAttribute(D, A, self.epsilon) # Race, Age
        # target_attr = self.chooseAttribute(D, A, self.epsilon) # Race, Age

        info = A[target_attr] # info[1] should hold the split value


        if self.feature_discrete[target_attr]:

            node = Node(feature_name=target_attr,  discrete=True,  depth=depth,  isLeaf=False, splitatr=target_attr)
            self.current_node = node

            tmp_D1 = D[np.argwhere( D[:, info[0]] == split_value), :]  # caphol
            tmp_D2 = D[np.argwhere( D[:, info[0]] != split_value ), :] # non-caphol

            keys = set(A.keys()).difference({target_attr}) 
            tmp_A = {key: A[key] for key in keys}          

            #print("A: ", A)
            print("tmp_D: ",split_value)            
                                                                              # tmp_A
            n1 = self.fit(tmp_D1.reshape((tmp_D1.shape[0], tmp_D1.shape[2])), tmp_A, depth + 1)
            n2 = self.fit(tmp_D2.reshape((tmp_D2.shape[0], tmp_D2.shape[2])), tmp_A, depth + 1)

            for possibleVal in info[1]:

                if possibleVal == split_value:
                    # tmp_D = tmp_D1
                    node[possibleVal] = n1
                    

                else:
                    # tmp_D = tmp_D2
                    node[possibleVal] = n2

                node.add_cat_child( possibleVal, node[possibleVal] )
            

        
        else:
       
            threshold = info[1]  # 'HoursPerWeek': [7, 48.764268089700906], 'Age': [0, 37.56920998722627],

            # target attr:  Age treshold:  28.5
            node = Node(feature_name=target_attr,  discrete=False,  threshold=threshold,  depth=depth,  isLeaf=False)

            self.current_node = node

            tmp_D = D[np.argwhere(D[:, info[0]] <= str(threshold)), :]

            node['<='] = self.fit(tmp_D.reshape(( tmp_D.shape[0], tmp_D.shape[2] )), A, depth+1)

            tmp_D = D[np.argwhere(D[:, info[0]]>str(threshold)), :]
            node['>'] = self.fit(tmp_D.reshape(( tmp_D.shape[0], tmp_D.shape[2] )), A, depth+1)

            node.set_left_child( node['<='] )
            node.set_right_child( node['>'] )
            
        return node
        

    def pred(self, D):
        return self.predict(D, self.A)
    
    def eval(self, D):
        return self.evaluate(D, self.A)


    def predict(self, D, A):

        #print("epslons spl: ", self.epsilon_sa )
        #print("epslons leafs: ", self.epsilon_l )
   
        row, _= D.shape # for the entire testing data 
        pred = np.empty((row, 1), dtype=str)

        pred_node = np.empty((row, 1), dtype=str)

        tmp_data={key: None for key in A.keys()}

        for i in range(len(D)):
            for key, info in A.items():

                tmp_data[key] = D[i, info[0]]

          
            pred[i] = self.tree(tmp_data) # -> calls __call__ in node -> result w/o noise

            fn = np.array([tmp_data[el] for el in tmp_data])

            tpr = self.tree.classify(fn, A, diction=True)

            pred_node[i] = tpr.noisy_label

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



class Node(object):

    def __init__(self, feature_name='', discrete=True, threshold=0, depth=0, isLeaf=False, splitatr=None, classification=None):

        self.map=dict()
        self.discrete=discrete
        self.splitatr=splitatr
        self.feature_name = feature_name

        self.isLeaf=isLeaf

        if isLeaf:
            self.classification=classification
        if not discrete:
            self.threshold=threshold

        # the class counts
        self._class_counts = defaultdict(int)

        # noise with exponential mechanism
        self._noisy_label = None

        # level
        self.depth=depth

        #cnildren
        self._left_child = None
        self._right_child = None
        self._cat_children = {}

    def __setitem__(self, key, value):
        self.map[key]=value

    def __getitem__(self, key):
        # this is get node
        return self.map.get(key)

    def __getdepth__(self):
        return self.depth 

    def increment_class_count(self, class_value):
        self._class_counts[str(class_value)] += 1  

    def is_leaf_check(self):
        return not self._left_child and not self._right_child and not self._cat_children

    def set_left_child(self, node):
        self._left_child = node

    def set_right_child(self, node):
        self._right_child = node

    def set_left_cat_child(self, cat, node):

        self._left_child = node

    def set_right_cat_child(self, cat, node):
        self._right_child = node

    def add_cat_child(self, cat_value, node):
        self._cat_children[str( cat_value )] = node

    @property
    def noisy_label(self):
        return self._noisy_label


    def set_noisy_label(self, 
                        epsilon, 
                        class_values):

 
        """Set the noisy label for this node"""
        if self.is_leaf_check():

            if not self._noisy_label:
                for val in class_values:
                    if val not in self._class_counts:
                        self._class_counts[val] = 0

                if max([v for k, v in self._class_counts.items()]) < 1:
                    self._noisy_label = np.random.choice([k for k, v in self._class_counts.items()])

                else:

                    utility = list( self._class_counts.values() ) # f_c -> frequency of the class count

                    candidates = list(self._class_counts.keys() )
         
                    mech = PermuteAndFlip(epsilon=epsilon, 
                                          sensitivity=1, # e^(-je) -> 1
                                          monotonic=True, 
                                          utility=utility,
                                          candidates=candidates)
                                          
                    self._noisy_label = mech.randomise()

        else:
            if self._left_child:
                self._left_child.set_noisy_label(epsilon, class_values)

            if self._right_child:
                self._right_child.set_noisy_label(epsilon, class_values)

            for child_node in self._cat_children.values():
                child_node.set_noisy_label(epsilon, class_values)

    def __call__(self, data): # returns the classification result! 
        
        '''
        This method should be used on the root to predict its classification.
        data: a dict with its key being the features and value being 
        the corresponding value.
        output: ?
        '''

        # print("self.map: ", self.map)

        if self.isLeaf:
            return self.classification

        if self.discrete:

            if data[self.feature_name] not in self.map.keys():
                
                nd = mode( self.map.values() )

                print("the map: ", self.map)
                print("data: ", data)

                print("feature ", self.feature_name, " not included as key in map keys.") # data["gill-col"]: "r" 
        
                print("data jolla tv korvataan: ", nd(data), type( nd(data)) )

                return nd(data)
    
            else:

                tv = self.map[data[self.feature_name]](data)

                return tv #self.map[data[self.feature_name]](data) # gets the correct node! -> shouldn't be an issue...

        else:

            if data[self.feature_name].astype(np.float32) > self.threshold:
                return self.map['>'](data)
            else:
                return self.map['<='](data)

    

    # new __call__ type method with child nodes as a part of the Node object!
    def classify(self, x, A, diction = False):
        """Classify the given data"""

        if self.is_leaf_check():
            #print("node for prediction: ")
            #print(self.__dict__)
            #print(" ")
            return self

        child = None

        ftn = A[self.feature_name][0] # ftn:  2

        if self.discrete: # feature IS discrete, therefore it is NOT CONTINUOUS

            # feature name in classify:  Education ->  fetch from self.A
            x_val = str(x[ftn])
            child = self._cat_children.get(x_val)

        else:

            x_val = x[ftn]

            if float(x_val) < self.threshold:
                child = self._left_child

            else:
                child = self._right_child

        if child is None:
         
            return self

        return child.classify(x, A)

     # wont be necessary if we find the correct leaf node with method classify!    
    def predict_with_node(self, X, A, classes):
        """Predict using this node orange"""
        y = []
        X = np.array(X)

        X = X[:,0:4]

        for x in X:

            node = self.classify(x, A) # !!!

            proba = np.zeros(len(classes)) #  [0. 0.]   ->  in other [0. 0.]

            cs = np.array(classes).astype(int)

            proba[np.where(cs == int(node.noisy_label))[0].item()] = 1

            y.append(proba)

        return np.array(y)