import numpy as np 
from collections import Counter, defaultdict
from diffprivlib.mechanisms import PermuteAndFlip

class Node(object):

    CONT_SPLIT = 0
    CAT_SPLIT = 1
    SNR = 0.0

    def __init__(self, feature_name='', discrete=True, threshold=0, depth=0, isLeaf=False, classification=None):

        self.map=dict()
        self.discrete=discrete
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

        # SNR
        self.SNR = 0.0

    def __setitem__(self, key, value):
        #print("set item used!set item used!
        # key: 0  value: <__main__.Node object at 0x7f5b7521b130>)
        # so set to key 0 a specific node...
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

    def add_cat_child(self, cat_value, node):
        self._cat_children[str(cat_value)] = node

    @property
    def noisy_label(self):
        return self._noisy_label


    def set_noisy_label(self, 
                        epsilon, 
                        class_values): # this is a list of strings! 

        #class values:  ['1', '0']
        #class counts:  defaultdict(<class 'int'>, {'0': 3})

        """Set the noisy label for this node"""
        if self.is_leaf_check():

            if not self._noisy_label:
                for val in class_values:
                    if val not in self._class_counts:
                        self._class_counts[val] = 0

                if max([v for k, v in self._class_counts.items()]) < 1:
                    self._noisy_label = np.random.choice([k for k, v in self._class_counts.items()])

                else:

                    #print("class counts: ")
                    #print(self._class_counts)

                    #print("noisy labels: ")
                    #print(self._noisy_label)

                    # class counts: 
                    #defaultdict(<class 'int'>, {'0': 2, '1': 0})

                    utility = list( self._class_counts.values() ) # f_c -> frequency of the class count

                    candidates = list(self._class_counts.keys())

                    print("Utility and epsilon: ")
                    print(utility, epsilon)

                    # Delta u : sensitivity
                  
                    mech = PermuteAndFlip(epsilon=epsilon, 
                                          sensitivity=1, # e^(-je) -> 1
                                          monotonic=True, 
                                          utility=utility,
                                          candidates=candidates)
                                          
                    self._noisy_label = mech.randomise()

                    print("noisy label after: ")
                    print(self._noisy_label)

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
        '''

        # print("data in call: ", data) # {'Age': '20', 'Education': '8', 'Occupation': '4', 'Gender': '1'}

        if self.isLeaf:
            return self.classification

        # if the node has a discrete (categorical) feature, use __getitem__ to find the next node
        # then call the next node.
        if self.discrete:

            # print("self.feature_name: ", self.feature_name) # Occupation
            # print(self.map) # {'0': <Node_1.Node object at 0x7f17608de1f0>, '10': <Node_1.Node object at 0x7f17608debe0>, '11': <Node_1.Node object at 0x7f17608de7c0>, ...

            return self.map[data[self.feature_name]](data) # gets the correct node! 

        else:

            if data[self.feature_name].astype(np.float32)>self.threshold:
                return self.map['>'](data)
            else:
                return self.map['<='](data)

    

    # new __call__ type method with child nodes as a part of the Node object!
    def classify(self, x, A, diction = False):
        """Classify the given data"""

        if self.is_leaf_check():
            return self

        child = None

        # data in classify:  ['57' '15' '10' '1']
        #attributes in classify:  {'Age': [0, 69.5], 'Education': [1, array(['0', '1', '10', '11', '12', '14', '15', '2', '4', '5', '6', '7',
        #'8', '9'], dtype='<U21')], 'Occupation': [2, array(['-1', '0', '10', '11', '12', '13', '2', '3', '4', '5', '6', '7',
        #'9'], dtype='<U21')], 'Gender': [3, array(['0', '1'], dtype='<U21')]}

        ftn = A[self.feature_name][0] # ftn:  2

        #else:

        #print("In Node classify")
        #print("x: ", x) 
        #print("ftn: ", ftn)

        #    print("datatype of x and x: ", x, type(x))
        if self.discrete: # feature IS discrete, therefore it is NOT CONTINUOUS

            # feature name in classify:  Education ->  fetch from self.A
            x_val = str(x[ftn])
            child = self._cat_children.get(x_val)

        else:

            x_val = x[ftn]

            if int(x_val) < self.threshold:
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

        # "classes" in prednode:  ['1', '0']   ->  in other [0 1], <class 'numpy.ndarray'>

        X = X[:,0:4]

        for x in X:

            node = self.classify(x, A) # !!!

            proba = np.zeros(len(classes)) #  [0. 0.]   ->  in other [0. 0.]

            # print("node noisy label: ", node.noisy_label)  1 -> in other 0

            #node:  <__main__.Node object at 0x7f52bca53e80>

            #classes in prednode:  ['1', '0']
            #proba:  [0. 0.]
            #node noisy label:  <bound method Node.noisy_label of <__main__.Node object at 0x7f52b7fd79d0>>
    

            #X in predict:  [['30' '0' '11' '1' '1']
            #['42' '15' '4' '1' '1']...
            #['56' '8' '2' '1' '0']] <class 'numpy.ndarray'>

            # [0. 0.] where  [1 0] is  1   -> OG: # [0. 0.] where [0 1] is 0 
            cs = np.array(classes).astype(int)

            proba[np.where(cs == int(node.noisy_label))[0].item()] = 1

            y.append(proba)

        return np.array(y)