# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np
import random


class Classifier:
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def fit(self, data, target):
        self.data = data
        self.target = target 
        self.random_forest = self.train_random_forest()
        self.evaluation_RF()

    def predict(self, features, legal=None):
       
        return  self.inference_RF(features)

    # decision tree Class 
    # Tree Data Structure
    class DecisionTree:
        def __init__(self,train_data, train_target, attribute_list):

            self.root = self.Node(train_data, train_target, attribute_list)

        def inference(self, input_data):

            return self.root.inference(input_data)

        #  A node Class in Decision Tree
        # If value of attribute = 1, go left
        # If value of attribute = 0, go right 
        class Node:
            def __init__(self,train_data, train_target, attribute_list):
                self.data = train_data
                self.target = train_target
                self.attributes = attribute_list
                self.left = None
                self.right = None
                self.best_attribute = -1 # -1 stand for this node is leaf Node
                self.train(train_data, train_target, attribute_list)
            
            # Calculate the gini index
            def calculate_gini(self,train_data, train_target, attribute_list):
                gini_index_list = []
                for attribute in attribute_list:
                    #only have 4 classes
                    T_counter = [0,0,0,0]
                    F_counter = [0,0,0,0]
                    #As all attributes are binary attributes (i.e. have value 0 or 1)
                    gini_T = 1
                    gini_F = 1

                    for index in range(len(train_data)):

                        class_belong = train_target[index]
                        if train_data[index][attribute] == 1:

                            T_counter[class_belong] += 1
                        else:
                            F_counter[class_belong] += 1
                    T_sum = sum(T_counter)
                    F_sum = sum(F_counter)

                    if T_sum != 0:
                        # Calculate the GINI for T

                        for number in T_counter:
                            gini_T -= (number/T_sum) ** 2
                    if F_sum != 0:
                        for number in F_counter:
                            gini_F -= (number/F_sum) ** 2
                    total = T_sum + F_sum
                    gini_index = (T_sum/total * gini_T) + (F_sum/total * gini_F)
                    gini_index_list.append(gini_index)
                return gini_index_list.index(min(gini_index_list))

            # The training algorithm for CART Decision Tree

            def train(self,train_data, train_target, attribute_list):

                # Check if all the data belongs to the same class

                if train_target[1:] == train_target[:-1]:
                    self.left = train_target[0]
                    self.right = train_target[0]
                    return
                
                # Check if the attribute set is empty
                if len(attribute_list) == 0:
                    # Set the leave node to the greatest common class
                    values, freq = np.unique(train_target, return_counts=True)
                    node_output = values[np.argmax(freq)]
                    self.left = node_output
                    self.right = node_output
                    return 

                else:
                    #Calculate the gini index of each attribute to determine the best one.
                    training_data = self.data
                    training_target = self.target
                    attributes = self.attributes
                    best_attribute_index = self.calculate_gini(training_data, training_target, attributes)
                    best_attribute = attributes[best_attribute_index]
                    self.best_attribute = best_attribute
                    # Extract the dataset for the sub-tree
                    true_samples = []
                    true_target = []
                    false_samples = []
                    false_target = []
                    for index in range(len(training_data)):
                        if training_data[index][best_attribute] == 1:
                            true_samples.append(training_data[index])
                            true_target.append(training_target[index])
                        else:
                            false_samples.append(training_data[index])
                            false_target.append(training_target[index])
                    
                    attributes.remove(best_attribute)
                    if len(true_samples) == 0:
                        values, freq = np.unique(train_target, return_counts=True)
                        node_output = values[np.argmax(freq)]
                        self.left = node_output
                    else:
                        leftNode = self.__class__(true_samples, true_target, attributes)
                        self.left = leftNode

                    if len(false_samples) == 0:
                        values, freq = np.unique(train_target, return_counts=True)
                        node_output = values[np.argmax(freq)]
                        self.right = node_output
                    else:
                        rightNode = self.__class__(false_samples, false_target, attributes)
                        self.right = rightNode
            
            def inference(self, input_data):
                # print(self.best_attribute)
                # If the left/right node is NOT a inference result, then do inference on the Node
                # Else return the inference result
                value = input_data[self.best_attribute]
                results = [0,1,2,3]
                if value == 1:
                    if self.left in results:
                        return self.left
                    else:
                        return self.left.inference(input_data)
                else:
                    if self.right in results:
                        return self.right
                    else:
                        return self.right.inference(input_data)   
        
            

    # Use bootstrapping to split the dataset
    def bootstrapping(self):
        num_of_sample = len(self.data)
        sample_data = []
        sample_target = []
        for i in range (num_of_sample):
            random_number = np.random.randint(num_of_sample, size = 1)[0]
            sample_data.append(self.data[random_number])
            sample_target.append(self.target[random_number])
        return sample_data, sample_target
    
    # Train the random forest
    # Return the list of decision trees in the forest
    def train_random_forest(self):
        tree_list = []
        total_attributes = len(self.data[0])
        attributes_list = []
        for index in range(total_attributes):
            attributes_list.append(index)
        # The recommend number of attributes for each weak Decision Tree is k=log2(d) 
        num_of_attribute_weak = int(round(np.log(total_attributes)/np.log(2)))
        #num_of_attribute_weak = 25

        # Number of tree in random forest
        num_of_tree = 128
        for i in range(num_of_tree):
            sample_data, sample_target = self.bootstrapping()
            sample_attribute = random.sample(attributes_list, num_of_attribute_weak)
            weak_tree = self.DecisionTree(sample_data, sample_target, sample_attribute)
            tree_list.append(weak_tree)
        return tree_list

    # Inference by the random forest
    # Plurality Voting
    # Input: feature of current state
    # Return: Best Action number of that state
    def inference_RF(self,state_array):
        results = []
        for tree in self.random_forest:

            result = tree.inference(state_array)
            results.append(result)

        values, freq = np.unique(results, return_counts=True)
        best_action = values[np.argmax(freq)]

        return best_action
    
    # Carry out the random forest evaluation.
    # Because the dataset size is limited, it is best to use all of the data so there is no overfitting in RF.
    def evaluation_RF(self):
        
        test_dataset = self.data
        test_target = self.target
        size_of_data = len(test_dataset)
        hit = 0
        for index in range(size_of_data):
            current_data = test_dataset[index]
            current_target = test_target[index]
            current_result = self.inference_RF(current_data)
            if current_result == current_target:
                hit += 1
        accuracy = float(hit)/float(size_of_data)
        print("The Accuracy is "+str(accuracy))