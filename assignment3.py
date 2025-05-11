import re
import math
import random

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Task 1.1
    # Return a unigram model.
    # Replace the line below with your code.

    # create a dictionary which will act as the value
    inner_dictionary = {}

    # calculate number of times each item shows up in sequence
    for item in sequence:
        if item not in inner_dictionary:
            # if the item does not exist in the inner dictionary add it and set frequency to 1
            inner_dictionary[item] = 1
        else:
            # otherwise item exists in the dictionary and add one to frequency
            inner_dictionary[item] += 1
                
    
    return {(): inner_dictionary}

def build_bigram(sequence):
    # Task 1.2
    # Return a bigram model.
    # Replace the line below with your code.

    # create a dictionary to hold values
    return_dictionary = {}

    # cycle through each item in sequency minus 1
    for i in range(len(sequence) - 1):
        # store the current item
        curr_item = (sequence[i],)


        # store the next item
        next_item = sequence[i + 1]

        # check to see if the item is not in the dictionary
        if curr_item not in return_dictionary:
            # add item to the dictionary
            return_dictionary[curr_item] = {}
        
        # if the next item has not already appeared add it to the current item's value
        if next_item not in return_dictionary[curr_item]:
            # add next item to current item's value and set frequency to 1
            return_dictionary[curr_item][next_item] = 1
        else:
            # otherwise add one to the frequency of the next item
            return_dictionary[curr_item][next_item] += 1
    
    # return the dictionary
    return return_dictionary 


def build_n_gram(sequence, n):
    # Task 1.3
    # Return an n-gram model.
    # Replace the line below with your code.

    # create the dictionary that will be returned
    return_dictionary = {}

    # for each item in sequence minus n plus one
    for i in range(len(sequence) - n + 1):
        # get the current items
        curr_items = tuple(sequence[i:i+n-1])

        # get the next item
        next_item = sequence[i + n - 1]
        
        # check to see if the current item is not in the dictionary
        if curr_items not in return_dictionary:
            # add it to the dictionary
            return_dictionary[curr_items] = {}
        
        # if the next item has not already appeared add it to the current item's value
        if next_item not in return_dictionary[curr_items]:
            # add next item to current item's value and set frequency to 1
            return_dictionary[curr_items][next_item] = 1
        else:
            # otherwise add one to the frequency of the next item
            return_dictionary[curr_items][next_item] += 1

    # return the dictionary
    return return_dictionary


def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    raise NotImplementedError

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor
    comp = 1 - factor
    for pred in preds[:-1]:
        if pred:
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            mult = comp * factor
            comp -= mult
    pred = preds[-1]
    mult += comp
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    
    print("task 1.1")
    model = build_unigram(sequence[:20])
    print(model)


    # Task 1.2 test code
    
    print("task 1.2")
    model = build_bigram(sequence[:20])
    print(model)
    

    # Task 1.3 test code

    print("task 1.3")
    model = build_n_gram(sequence[:20], 5)
    print(model)
    

    # Task 2 test code
    '''
    print(query_n_gram(model, tuple(sequence[:4])))
    '''

    # Task 3 test code
    '''
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    '''

    # Task 4.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 4.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''

