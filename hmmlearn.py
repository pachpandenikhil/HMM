import json, sys, io
from itertools import tee

# Returns contiguous pair-wise elements from list of elements
# list : list of elements
# input: [1,2,3,...]
# output: [(1,2),(2,3),....]
def pairwise(list):
    x, y = tee(list)
    next(y, None)
    return zip(x, y)


# Increments the count of word in the emissions count map of the tag
# If the word does not exist, new entry is added to the amp with count = 1
def update_tag_emission_count(tag_emission_count, word):
    if word in tag_emission_count:
        tag_emission_count[word] += 1
    else:
        tag_emission_count[word] = 1
    return tag_emission_count


# Reads the training corpus and returns:
# line_count: tot number of tagged sentences in the corpus
# tags: universal set of tags in the corpus
# start_count: map of tag and count of sentences where it occurred at the start of the sentence
# transition_count: map of transition counts between tags
# emission_count: map of tag and its emission counts
# tag_count: map of tag and its tot count in the corpus
def read_train_corpus(train_corpus_file):
    # Set of all tags
    tags = set()

    # Map for start counts
    start_count = {}

    # Map for transition counts
    transition_count = {}

    # Map for emission counts
    emission_count = {}

    # Map for total tag counts
    tag_count = {}

    f = open(train_corpus_file)
    line_count = 0
    for line in iter(f):
        line = line.rstrip()
        if line:
            line_count += 1
            tokens = line.split(' ')

            # Adding start count for first tag
            first_token = tokens[0]
            first_tag = first_token[-2:]
            first_word = first_token[0:-3]
            if first_tag in start_count:
                start_count[first_tag] += 1
            else:
                start_count[first_tag] = 1

            # Adding tag count for first tag
            if first_tag in tag_count:
                tag_count[first_tag] += 1
            else:
                tag_count[first_tag] = 1

            # adding emission count for first tag
            first_tag_emission_count = {}
            if first_tag in emission_count:
                first_tag_emission_count = emission_count[first_tag]
            first_tag_emission_count = update_tag_emission_count(first_tag_emission_count, first_word)
            emission_count[first_tag] = first_tag_emission_count

            # Adding transition and emission counts
            for src_token, dest_token in pairwise(tokens):

                src_tag = src_token[-2:]
                dest_tag = dest_token[-2:]
                dest_word = dest_token[0:-3]

                tags.add(src_tag)
                tags.add(dest_tag)

                # adding transition counts
                if src_tag in transition_count:
                    src_tag_transition_map = transition_count[src_tag]
                    if dest_tag in src_tag_transition_map:
                        src_tag_transition_map[dest_tag] += 1
                    else:
                        src_tag_transition_map[dest_tag] = 1
                    transition_count[src_tag] = src_tag_transition_map
                else:
                    src_tag_transition_map = {}
                    src_tag_transition_map[dest_tag] = 1
                    transition_count[src_tag] = src_tag_transition_map

                # adding emission counts
                dest_tag_emission_count = {}

                if dest_tag in emission_count:
                    dest_tag_emission_count = emission_count[dest_tag]

                dest_tag_emission_count = update_tag_emission_count(dest_tag_emission_count, dest_word)
                emission_count[dest_tag] = dest_tag_emission_count

                # adding tag count
                if dest_tag in tag_count:
                    tag_count[dest_tag] += 1
                else:
                    tag_count[dest_tag] = 1
    f.close()
    return line_count, tags, start_count, transition_count, emission_count, tag_count


# Returns the map of start probabilities from start counts
def get_start_probability(start_count, tags, train_data_size):
    start_probability = {}
    vocabulary_size = len(tags)
    smoothing_denominator = train_data_size + vocabulary_size

    # computing probabilities with add-one smoothing
    for tag in tags:
        tag_count = 1
        if tag in start_count:
            tag_count += start_count[tag]
        start_probability[tag] = float(tag_count) / smoothing_denominator
    return start_probability


# Returns the map of transition probabilities for a tag from its transition counts to other tags
def get_tag_transition_probability(tag_transition_count, tags):
    tag_transition_probability = {}
    vocabulary_size = len(tags)
    tag_count = 0

    for k, v in tag_transition_count.items():
        tag_count += v

    smoothing_denominator = tag_count + vocabulary_size

    for tag in tags:
        smoothing_numerator = 1
        if tag in tag_transition_count:
            smoothing_numerator += tag_transition_count[tag]
        tag_transition_probability[tag] = float(smoothing_numerator)/smoothing_denominator

    return tag_transition_probability


# Returns the map of transition probabilities from transition counts between tags
def get_transition_probability(transition_count, tags):
    transition_probability = {}
    for tag in tags:
        tag_transition_probability = {}
        tag_transition_count = {}
        if tag in transition_count:
            tag_transition_count = transition_count[tag]
        tag_transition_probability = get_tag_transition_probability(tag_transition_count, tags)
        transition_probability[tag] = tag_transition_probability
    return transition_probability


# Returns the map of emission probabilities from emission counts and tot tag counts from the corpus
def get_emission_probability(emission_count, tag_count):
    emission_probability = {}
    for tag, tag_emission_count in emission_count.items():
        tag_emission_probability = {}
        for word, word_count in tag_emission_count.items():
            word_probability = float(word_count)/tag_count[tag]
            tag_emission_probability[word] = word_probability
        emission_probability[tag] = tag_emission_probability
    return emission_probability


# Writes the model to file in Unicode format
def write_model(model, filePath):
    with io.open(filePath, 'w', encoding='utf8') as model_file:
        data = json.dumps(model, indent=4, ensure_ascii=False, encoding='utf8')
        model_file.write(data)

# main execution
if __name__ == '__main__':

    # reading training corpus
    train_corpus_file = sys.argv[1]
    train_data_size, tags, start_count, transition_count, emission_count, tag_count = read_train_corpus(train_corpus_file)

    # computing probabilities
    start_probability = get_start_probability(start_count, tags, train_data_size)
    transition_probability = get_transition_probability(transition_count, tags)
    emission_probability = get_emission_probability(emission_count, tag_count)

    # creating data structure for writing the model to file
    model = {}
    model["start_probability"] = start_probability
    model["transition_probability"] = transition_probability
    model["emission_probability"] = emission_probability

    # writing the model to file
    write_model(model, "hmmmodel.txt")