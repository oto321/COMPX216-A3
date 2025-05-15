import re
import math
import random

def tokenise(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
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
        curr_items = tuple  (sequence[i:i+n-1])

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

    # If it's a unigram model, return the dictionary at key ()
    if () in model:
        return model[()]
    # For n-gram models where n >= 2
    return model.get(sequence, None)

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

    # list to store predictins
    predictions = []

    # for each model in models
    for model in models:
        # get size of context in model
        size = len(next(iter(model)))

        # check to see if the sequence is long enough for the n gram
        if len(sequence) >= size:
            
            # retrieve size number of last words from the sequence
            value = tuple(sequence[-size: ])

            # call query_n_gram to get all the possible next words
            # and add it to the predictions list if it exists 
            prediction = query_n_gram(model, value)
            predictions.append(prediction) if prediction is not None else None 
            

    # use the blended_probabilities function
    blended_predictions = blended_probabilities(predictions)

    # return a random word from the blended_next_words
    return random.choices(list(blended_predictions.keys()), weights=blended_predictions.values(), k=1)[0]



def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.

    # store the total likelihood
    total_likelihood = 0

    # get the context size of the first model to figure out what the 
    max_n_val = len(next(iter(models[0]))) + 1

    # for each token in the sequence
    for i, token in enumerate(sequence):

        # determine what model to use
        n_value = min(i, len(models) - 1)
        
        curr_model = models[-(n_value+1)]


        # get the context
        context = tuple(sequence[i - n_value : i])

        # get a list of all the predictions using the current mode
        predictions_list = query_n_gram(curr_model, context)

        # check to see if the prediction exists
        if predictions_list is None or token not in predictions_list:
            return -math.inf

        # get the frequency of the current token
        curr_token_frequency = predictions_list[token]

        # get the total number of predictions in the predictions list
        predictions_total = sum(predictions_list.values())

        # probability of the current token by normalizing the frequency of the current word
        probability = curr_token_frequency/predictions_total

        # get the log of it
        log_probability = math.log(probability)

        # add it to the total
        total_likelihood += log_probability

    return total_likelihood





def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    
    total_score = 0

    # for each token in the sequence
    for i, token in enumerate(sequence):

        # create a list to store the predictions
        predictions = []

        # for each model in the models
        for model in models:

            # get the size of the current model
            curr_model_size = len(next(iter(model)))
            
            # check to see if the number of tokens before the current 
            # token is more than or equal to the size of the current mode
            if i >= curr_model_size:

                # get the context
                context = tuple(sequence[i - curr_model_size: i])

                # get the prediction
                prediction = query_n_gram(model, context)

                # add the prediction to predictions if it is not none
                predictions.append(prediction) if prediction is not None else None 

        # if there are no predictions return inf
        if not predictions:
            return -math.inf


        blended = blended_probabilities(predictions)

        # If token not found in blended predictions, prob is 0
        if token not in blended or blended[token] == 0:
            return -math.inf

        # Add log probability to total
        total_score += math.log(blended[token])

    return total_score

    


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
    
    print("task 2")
    print(query_n_gram(model, tuple(sequence[:4])))
    

    # Task 3 test code
    
    print("task 3")
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    

    # Task 4.1 test code
    
    print("task 4.1")
    print(log_likelihood_ramp_up(sequence[:20], models))
    

    # Task 4.2 test code
    
    print("task 4.2")
    print(log_likelihood_blended(sequence[:20], models))
    

