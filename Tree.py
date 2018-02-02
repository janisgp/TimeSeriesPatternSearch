'''
Created on 10 Jan 2018

@author: Janis
'''

import numpy as np
from matplotlib import pyplot as plt

class DecisionTree:
    def __init__(self,data,depth,features_names,type):
        
        self.class_column = data.shape[1] - 1
        self.hits = np.sum(data[:,self.class_column]) # number of targets with class 1
        self.type = type    # type: so far features or pattern
        
        # List for storing high value paths
        self.high_value_paths = []
        
        #initialize root element
        classes, counts = np.unique(data[:,self.class_column],return_counts=True)
        gini = computeGini(data[:,self.class_column])
        rootNode = DecisionNode(None,classes,counts,gini,self.hits)
        self.root = rootNode
        
        self.features_names = features_names # ordered list with names of the features
        
        depthCounter = 0 #counting the depth of the tree
        #call growTree recursively
        self.growTree(self.root, data, depth, depthCounter)
        
    def growTree(self, Node,  data, depth, depthCounter):
        
        if self.type == 'features':
            data1, data2, threshold = findBestSplit_continuous(data,self.class_column) #compute best split
        if self.type == 'pattern':
            data1, data2, threshold = findBestSplit_discrete(data,self.class_column) #compute best split
        
        if type(data1).__module__ == np.__name__: #check if the is a good split
            #is the max depth reached or whether one of the sub arrays has 
            #reached the limit
            if depthCounter == depth or Node.gini == 0: 
                return -1
            
            #increment depth counter, not a leaf, setting threshold
            nextDepthCounter = depthCounter + 1
            Node.leaf = False
            Node.setThreshold(threshold)
            
            #check whether subarry has more than one row and then grow subtree
            if data1.shape[0]>1:
                classes, counts = np.unique(data1[:,self.class_column],return_counts=True)
                gini = computeGini(data1[:,self.class_column])
                newNode = DecisionNode(Node, classes, counts, gini, self.hits)
                newNode.setLarger(False)
                self.growTree(newNode, data1, depth, nextDepthCounter)
                Node.appendLeftChild(newNode)
            if data2.shape[0]>1:
                classes, counts = np.unique(data2[:,self.class_column],return_counts=True)
                gini = computeGini(data2[:,self.class_column])
                newNode = DecisionNode(Node, classes, counts, gini, self.hits)
                newNode.setLarger(True)
                self.growTree(newNode, data2, depth, nextDepthCounter)
                Node.appendRightChild(newNode)
        else:
            return -1
    
    def print_tree(self):
        lvl = 0
        self.print_nodes(self.root, lvl)
        return
     
    def print_nodes(self, node, lvl):
        print('Node level ' + str(lvl))
        if node.leaf:
            print('Score: ' + str(node.score) + ' ||| Purity: ' + str(node.purity) + ' ||| Coverage: ' + str(node.coverage) + ' ||| Classes: ' + str(node.classes) + ' ||| Distribution: ' + str(node.distribution) + ' ||| Gini: ' + str(node.gini))
        else: 
            print('Score: ' + str(node.score) + ' ||| Purity: ' + str(node.purity) + ' ||| Coverage: ' + str(node.coverage) + ' ||| Classes: ' + str(node.classes) + ' ||| Distribution: ' + str(node.distribution) + ' ||| Gini: ' + str(node.gini) + ' ||| Threshold: ' + 
                  str(node.threshold) + ' ||| corresponding to feature: ' + self.features_names[node.threshold[0]])
        lvl+=1
        if node.leftChild != None:
            self.print_nodes(node.leftChild, lvl)
        if node.rightChild != None:
            self.print_nodes(node.rightChild,lvl)
        lvl-=1
        return
       
    def classifyPoint(self, point):
        #classifies a point according to grown tree
        
        #walking the tree down towards the suitable leaf
        currentNode = self.root
        while not currentNode.leaf:
            if point[currentNode.threshold[0]] <= currentNode.threshold[1]:
                currentNode = currentNode.leftChild
            elif point[currentNode.threshold[0]] > currentNode.threshold[1]:
                currentNode = currentNode.rightChild
        
        probability = np.max(currentNode.distribution)/np.sum(currentNode.distribution) #probability 
        classification = currentNode.classes[np.argmax(currentNode.distribution)] #classification
        return classification, probability
    
    def returnPaths(self, purity_threshold, coverage_threshold):
        # prints the characteristics of paths to intersting nodes
        
        current_node = self.root
        self.exploreNode(current_node, purity_threshold, coverage_threshold)
        
        return self.high_value_paths
            
    def exploreNode(self, node, purity_threshold, coverage_threshold):
        if node.leaf:
            if node.purity >= purity_threshold and node.coverage >= coverage_threshold:
                
                discr_features_string = []      # store discriminating features
                discr_features = []
                current_node = node
                while current_node.parent != None:
                    
                    current_discr = {
                        'feature' : None,
                        'pattern' : None,
                        'score' : current_node.score,
                        'purity' : current_node.purity,
                        'coverage' : current_node.coverage,
                        'in' : not current_node.larger
                        }
                    
                    comp = '' # for writing output string
                    if current_node.larger:
                        comp = 'larger'
                    else:
                        comp = 'smaller'
                    
                    current_discr['feature'] = current_node.parent.threshold[0]
                    current_discr['pattern'] = current_node.parent.threshold[1]
                    
                    discr_features_string.append('Score: ' + str(current_node.score) + ' ||| Purity: ' + str(current_node.purity) + ' ||| Coverage: ' + str(current_node.coverage) + ' ||| Last Split at feature: ' +
                                          self.features_names[current_node.parent.threshold[0]] + ' ; this nodes elements are ' + comp + ' than ' + 
                                          str(current_node.parent.threshold[1]))
                    
                    discr_features.append(current_discr)
                    current_node = current_node.parent
                    
                print('### PRINTING HIGH VALUE NODE ###')
                for node_info in reversed(discr_features_string):
                    print(node_info)
                
                self.high_value_paths.append(discr_features)
        
        if not node.leaf:
            if node.leftChild != None:
                self.exploreNode(node.leftChild, purity_threshold, coverage_threshold)
            if node.rightChild != None:
                self.exploreNode(node.rightChild, purity_threshold, coverage_threshold)
              
                    
        
class DecisionNode:
    def __init__(self, parent, classes, distribution, gini, hits):
        #self.threshold = threshold
        self. parent = parent
        self.classes = classes
        self.distribution = distribution
        self.gini = gini
        self.leftChild = None
        self.rightChild = None
        self.leaf = True
        
        # compute purity and coverage at this node
        if self.classes.shape[0] == 1:
            if self.classes[0] == 1:
                self.purity = 1.0
                self.coverage = distribution[0] / hits
            else:
                self.purity = 0
                self.coverage = 0
        else: 
            self.purity = self.distribution[1]/np.sum(self.distribution)
            self.coverage = self.distribution[1]/hits
            
        # compute score: score = ((100*delta pur)^2 + (100*delta cov)^2)*cov
        # delta compared to parent!
        if parent != None:
            dpur = self.purity - self.parent.purity
            dcov = self.coverage - self.parent.coverage
            self.score = (np.sign(dpur)*(100*dpur)**2 + np.sign(dcov)*(100*dcov)**2)*self.coverage
        else: 
            self.score = 0
        
    
    def setThreshold(self, threshold):
        self.threshold = threshold
    
    def appendLeftChild(self, child):
        self.leftChild = child
    
    def appendRightChild(self, child):
        self.rightChild = child
        
    def setLarger(self,larger):
        # variable is needed in order to determine whether the data at this point
        # is smaller or larger than the threshold of the parent
        self.larger = larger
        
        
def findBestSplit_continuous(data, class_column):
    #returns split, feature (0,1,2) and threshold 
    
    #compute array of gini indeces and corresponding threshold. 
    #The feature is implicitly stored in the second dimension
    currentGini = computeGini(data[:,class_column])
    number_features = data.shape[1]-1
    N = data.shape[0]
    giniArray = np.zeros([number_features,N-1,2])
    for i in range(number_features):
        data =  data[data[:,i].argsort()]
        for j in range(data[:,i].size - 1):
            currentThreshold = (data[j,i] + data[j+1,i])/2
            gini1 = computeGini(data[0:j+1,class_column])
            gini2 = computeGini(data[j+1:,class_column])
            #inlcuding the probability that you end up in the subset
            giniSum = (data[0:j+1,class_column].shape[0]/N)*gini1 + \
                (data[j+1:,class_column].shape[0]/N)*gini2
            giniArray[i,j,0] = currentThreshold
            giniArray[i,j,1] = giniSum
    
    #get feature, threshold and row index (latter in sorted array!)
    #until the row index inclusively!!!
    row = np.argmin(giniArray[:,:,1])%giniArray.shape[1]
    feature = np.argmin(giniArray[:,row,1])
    threshold = giniArray[feature,row,0]
    
    gini_split = giniArray[feature,row,1]
    
    #return -1 if split is worse than current status
    if (currentGini - gini_split) <= 0:
        return None,None,[None,None]

    #get the optimal split
    data =  data[data[:,feature].argsort()] #sort again
    data1 = data[0:row+1,:]
    data2 = data[row+1:,:]
    
    return data1, data2, [feature,threshold]

def findBestSplit_discrete(data, class_column):
    #returns split, feature (0,1,2) and threshold 
    
    #compute array of gini indeces and corresponding threshold. 
    #The feature is implicitly stored in the second dimension
    currentGini = computeGini(data[:,class_column])
    number_features = data.shape[1]-1
    N = data.shape[0]
#     giniArray = np.ones([number_features,N-1,2])
    
    best_gini = 1
    best_feature = -1
    best_pattern = -1
    
    for i in range(number_features):
        
        unique_patterns =  np.unique(data[:,i])
        j = 0
        for pattern in unique_patterns:
            mask = data[:,i] == pattern
            mask_rev = data[:,i] != pattern
            gini1 = computeGini(data[mask,class_column])
            gini2 = computeGini(data[mask_rev,class_column])
            #inlcuding the probability that you end up in the subset
            giniSum = (data[mask,class_column].shape[0]/N)*gini1 + \
                (data[mask_rev,class_column].shape[0]/N)*gini2
            
            if giniSum < best_gini:
                best_gini = giniSum
                best_feature = i
                best_pattern = pattern
            
            j += 1
    
    #get feature, threshold and row index (latter in sorted array!)
    #until the row index inclusively!!!
    
    #return -1 if split is worse than current status
    if (currentGini - best_gini) <= 0:
        return None,None,[None,None]

    #get the optimal split
    mask = data[:,best_feature] == best_pattern
    mask_rev = data[:,best_feature] != best_pattern
    data1 = data[mask,:]
    data2 = data[mask_rev,:]
    
    return data1, data2, [best_feature,best_pattern]

def computeGini(dataArray):
    #computing the gini index for a class vector dataArray
    
    #get an array with the counts of the different classes and the number of elements
    _, counts = np.unique(dataArray,return_counts=True)
    elementCount = np.sum(counts)
    #compute gini
    giniIndex = 1
    for count in counts:
        giniIndex -= (count/elementCount)**2
    return giniIndex


