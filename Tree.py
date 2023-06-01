import numpy as np 
from Node import Node
import random
from collections import Counter, defaultdict

class Tree_DPDT():

    '''
    The main class of decision tree.
    '''
    def __init__(self, A_ind, A, attribute_values, root, class_values, feature_discrete, treetype, dataset_size, epsilon_per_tree, max_depth):
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


    def randomOrder(self, D, A):

        '''
        Return the order by Random choise
        For the definition of D and A, see the remark in method 'fit'.
        '''

        tmp_value_dict=dict()
        med_val_dict=dict()

        for attr, info in A.items():

            # all the attributes possible values
            possibleVal = np.unique(D[:, info[0]])  #info[0] is the index to the array column where possible values of attribute are in D
            # this should not affect sensitivity...

            # if the continuous attribute have only one possible value, then 
            # choosing it won't improve the model, so we abandon it.
            if len(possibleVal)==1:
                continue

            if self.feature_discrete[attr] is True:

                # discrete
                if len(info) < 2:
                    A[attr].append(possibleVal)

                # A:  {'Gender': [3, array(['0', '1'], dtype='<U21')], 'Age': [0, 25.5], 'Education': [1, array(['0', '1', '10', '11', '12', '14', '15', '2', '4', '5', '6', '7',
                # '8', '9'], dtype='<U21')]}

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

                    if IC_tmp>maxMetric:
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


        #let_to_int = dict(sorted(let_to_int.items()))
        
        attr_list=list(tmp_value_dict.keys())
        
        #print("attrlist b4", attr_list)

        attr_list.sort(key=lambda x: tmp_value_dict[x])

        #print("attrlist aftr", attr_list)

        return attr_list

    def medianSplitOrder(self, D, A):

        '''
        Return the order by Random choise
        For the definition of D and A, see the remark in method 'fit'.
        '''

        tmp_value_dict=dict()

        for attr, info in A.items():

            # all the attributes possible values
            possibleVal = np.unique(D[:, info[0]])  #info[0] is the index to the array column where possible values of attribute are in D
            # this should not affect sensitivity...

            # if the continuous attribute have only one possible value, then 
            # choosing it won't improve the model, so we abandon it.
            if len(possibleVal)==1:
                continue

            if self.feature_discrete[attr] is True:

                # discrete
                if len(info) < 2:
                    A[attr].append(possibleVal)

                # A:  {'Gender': [3, array(['0', '1'], dtype='<U21')], 'Age': [0, 25.5], 'Education': [1, array(['0', '1', '10', '11', '12', '14', '15', '2', '4', '5', '6', '7',
                # '8', '9'], dtype='<U21')]}

                

                # random value between 0 and 1
                IC_value = random.uniform(0, 1) # assign random 'values' to the attributes
                # in here the scores with exponential mechanism will be calculated... 

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

                    if IC_tmp>maxMetric:
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
        attr_list.sort(key=lambda x: tmp_value_dict[x]) # in here the MSE will be conducted

        return attr_list

    def chooseAttribute(self, D, A, eps):

        # print("epsilon passed: ", eps)

        if self.treeType=='ID3':
            attr_list=self.orderByGainOrRatio(D, A, by='Gain')
            return attr_list[-1]

        if self.treeType=='C4.5':
            attr_list=self.orderByGainOrRatio(D, A, by='Gain')

            # for C4.5, we choose the attributes whose Gain are above average
            # and then order them by Ratio.

            sub_A={key: A[key] for key in attr_list}
            attr_list=self.orderByGainOrRatio(D, sub_A, by='Ratio')

            return attr_list[-1]

        if self.treeType == 'Random':

            attr_list = self.randomOrder(D, A)

            print("chosen attr: ", attr_list[-1])

            return attr_list[-1]

        if self.treeType == 'Median':

            split_list = self.medianSplitOrder(D, A)

            return attr_list[-1]

    def train(self, D, depth):

        X = D[:, 0:D.shape[1]-1] # datarows
        y = D[:, -1] # targets

        X = np.array(X)

        y = [[el] for el in y]
        y = np.array(y)
        
        self.tree = self.fit(D, self.A, depth)

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
        target_attr = self.chooseAttribute(D, A, self.budget/2)

        # print("target_attr: ", target_attr)


        # generate nodes for each possible value of the target attribute if it's discrete
        # related information is stored in A[target_attr][1] now, 
        # since we have called chooseAttribute at least once.
        # "divide the current node to MULTIPLE CHILD NODES according to class labels
        # a new node
        
        info = A[target_attr]


        if self.feature_discrete[target_attr]:

            node = Node(feature_name=target_attr, 
                        discrete=True, 
                        depth=depth, 
                        isLeaf=False)


            self.current_node = node

            # generate nodes for each possible value

            # print("info1: ", info[1]) 
            # info1:  ['-1' '0' '10' '11' '12' '13' '2' '3' '4' '5' '6' '7' '9']


            for possibleVal in info[1]:

                # important, for this affects tmp_A
                # keys is just the names of the attributes ("Age", "Gender" etc.) without the target attr name.
                keys=set(A.keys()).difference({target_attr})
                # print("Attributes: ")
                # print(A) # -> problem in A

                # connect node to its child
                tmp_D = D[np.argwhere(D[:, info[0]]==possibleVal), :]
                
                tmp_A = {key: A[key] for key in keys}
                
                # this here calls  def __setitem__(self, key, value):

                #print("given as input to node[possibleVal] ")
                #print("reshaper: ")
                #print(  tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2]))  )
                # print("tmp_A: ")
                # print(tmp_A)

                #reshaper: 
                #[['37' '5' '11' '0' '0' '4' '4' '12' '0' '1']
                #...
                #['62' '2' '15' '0' '0' '4' '4' '30' '0' '1']]
                
                #tmp_A: 
                #{'Gender': [8, array(['0', '1'], dtype='<U21')], 
                # 'HoursPerWeek': [7, 48.764268089700906], 
                # 'Race': [6, array(['0', '1', '2', '3', '4'], dtype='<U21')], 
                # 'Workclass': [1, array(['0', '1', '2', '3', '4', '5'], dtype='<U21')], 
                # 'Relationship': [5, array(['0', '1', '2', '3', '4', '5'], dtype='<U21')], 
                # 'Age': [0, 37.56920998722627], 
                # 'Education': [2, array(['0', '1', '10', '11', '12', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U21')]}

                node[possibleVal] = self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), tmp_A, depth + 1)
               
                # added to discrete children
                # this here calls  def __getitem__():
                node.add_cat_child(possibleVal, node[possibleVal])

        
        else:
            # generate two nodes for the two classification if it's continuous
            # continuous

            threshold = info[1]

            # treshold_valie = np.random.uniform()

            #print("domains in self.attribute_values")
            #print(self.attribute_values)
            #print("target attr")

            # confused over how the split value gets found during the continuous attribute node split...

            # target attr:  Age treshold:  28.5
            node=Node(feature_name=target_attr, 
                      discrete=False, 
                      threshold=threshold, 
                      depth=depth, 
                      isLeaf=False)

            self.current_node = node

            tmp_D=D[np.argwhere(D[:, info[0]]<=str(threshold)), :]
            node['<='] = self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A, depth+1)

            tmp_D=D[np.argwhere(D[:, info[0]]>str(threshold)), :]
            node['>'] = self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A, depth+1)

            node.set_left_child( node['<='] )
            node.set_right_child( node['>'] )
            
        
        return node


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
        tmp_node = Node(feature_name='leaf-'+str(self.leaf_count), isLeaf = True, classification=most_frequent)
        


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

        # print("the tree is a: ", type(self.tree))
        # the tree is a:  <class '__main__.Node'>

        for i in range(len(D)): # only 1 row?
            for key, info in A.items():

                tmp_data[key] = D[i, info[0]]

            # print("data given to tree: ")
            # print(tmp_data)
            # {'Age': '20', 'Education': '8', 'Occupation': '4', 'Gender': '1'}

            # but self.tree is evidently initialized as none?

            pred[i] = self.tree(tmp_data) # -> calls __call__ in node

            # node = self.tree.classify(X[i], self.A)
            # datatype of x and x:  ['57' '15' '10' '1'] <class 'numpy.ndarray' 
            # datatype of fn and fn:  dict_values(['23', '15', '7', '1']) <class 'numpy.ndarray'>
            fn = np.array([tmp_data[el] for el in tmp_data])

            pred_node[i] = self.tree.classify(fn, A, diction=True).noisy_label

            # print("pred[i] in pred loop: ", pred[i]) ['1'], ['0']
      
        return pred_node
    
    def evaluate(self, testing_D, A):
       
        true_label = testing_D[:, -1]
        pred_label = self.predict(testing_D, A)
        
        success_count=0
        for i in range(len(true_label)):
            if true_label[i]==pred_label[i]:
                success_count+=1

        return success_count/len(true_label)