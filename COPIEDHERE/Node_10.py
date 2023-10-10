import numpy as np 
from collections import Counter, defaultdict
from diffprivlib.mechanisms import PermuteAndFlip

class Node(object):

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

    def add_cat_child(self, cat_value, node):
        self._cat_children[str( cat_value )] = node

    @property
    def noisy_label(self):
        return self._noisy_label


    def set_noisy_label(self, 
                        epsilon, 
                        class_values): # this is a list of strings! 

 
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

                    candidates = list(self._class_counts.keys())
         
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
        '''

        print("data in call: ", data) # {'Age': '20', 'Education': '8', 'Occupation': '4', 'Gender': '1'}

        if self.isLeaf:
            return self.classification

        if self.discrete:

            print("self.feature_name: ", self.feature_name) # Occupation
            print(self.map)

            return self.map[data[self.feature_name]](data) # gets the correct node! -> shouldn't be an issue...

        else:

            if data[self.feature_name].astype(np.float32) > self.threshold:
                return self.map['>'](data)
            else:
                return self.map['<='](data)

    

    # new __call__ type method with child nodes as a part of the Node object!
    def classify(self, x, A, diction = False):
        """Classify the given data"""

        if self.is_leaf_check():
            return self

        child = None

        ftn = A[self.feature_name][0] # ftn:  2

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

        X = X[:,0:4]

        for x in X:

            node = self.classify(x, A) # !!!

            proba = np.zeros(len(classes)) #  [0. 0.]   ->  in other [0. 0.]

            cs = np.array(classes).astype(int)

            proba[np.where(cs == int(node.noisy_label))[0].item()] = 1

            y.append(proba)

        return np.array(y)