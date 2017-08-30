import json, ast, sys, math

# Returns the emission probability of the word for the specified tag
def get_word_tag_emission_probability(word, tag, emission_probability):
    word_tag_emission_probability = 0
    if word in emission_probability[tag]:
        word_tag_emission_probability = emission_probability[tag][word]

    return word_tag_emission_probability


# Returns math.log() for the specified probability
# Returns negligible value(math.log(0.0000000000000001)) for 0 probabilities
def get_log(probability):
    if probability == 0.0:
        return MIN_LOG_VALUE
    else:
        return math.log(probability)


# Viterbi algorithm implementation
# Uses log operations to avoid decimal underflow
def viterbi(observations, tags, start_probability, trans_probability, emit_probability):
    viterbi = [{}]
    # calculating probabilities for first transition
    for tag in tags:
        viterbi[0][tag] = {"prob": get_log(start_probability[tag]) + get_log(get_word_tag_emission_probability(observations[0], tag, emit_probability)),
                    "prev": None}

    # calculating max probability and backpointers for each step
    for idx in range(1, len(observations)):
        viterbi.append({})
        for tag in tags:
            max_tr_prob = max(viterbi[idx - 1][prev_tag]["prob"] + get_log(trans_probability[prev_tag][tag]) for prev_tag in tags)
            for prev_st in tags:
                if viterbi[idx - 1][prev_st]["prob"] + get_log(trans_probability[prev_st][tag]) == max_tr_prob:
                    max_prob = max_tr_prob + get_log(get_word_tag_emission_probability(observations[idx], tag, emit_probability))
                    viterbi[idx][tag] = {"prob": max_prob, "prev": prev_st}
                    break
    output = []

    max_prob = max(value["prob"] for value in viterbi[-1].values())
    previous = None

    # finding the most probable state and its backtrack
    for tag, data in viterbi[-1].items():
        if data["prob"] == max_prob:
            output.append(tag)
            previous = tag
            break

    # following the backpointers to determine the POS sequence
    for idx in range(len(viterbi) - 2, -1, -1):
        output.insert(0, viterbi[idx + 1][previous]["prev"])
        previous = viterbi[idx + 1][previous]["prev"]

    return output


def is_unseen_word(word, emission_probability):
    is_unseen_word = True
    for tag, tag_emission_probability in emission_probability.items():
        if word in tag_emission_probability:
            is_unseen_word = False
            break
    return is_unseen_word


# Returns the map of unseen observations
def get_unseen_observations(observations, emission_probability):
    unseen_observations = {}
    for word in observations:
        if is_unseen_word(word, emission_probability):
            unseen_observations[word] = 1
    return unseen_observations


# Adds unseen observations as proper nouns with emission probability of 1
def add_unseen_observations(emission_probability, unseen_observations):
    np_emissions = emission_probability["NP"]
    np_emissions.update(unseen_observations)
    emission_probability["NP"] = np_emissions
    return emission_probability


# Executes Viterbi algorithm for every test sentence and generates the output
def execute_viterbi(start_probability, transition_probability, emission_probability, test_file):
    output_lines = []
    tags = tuple(transition_probability.keys())
    f = open(test_file)
    for line in iter(f):
        line = line.rstrip()
        if line:
            words = line.split(' ')
            observations = tuple(words)
            unseen_observations = get_unseen_observations(observations, emission_probability)
            emission_probability = add_unseen_observations(emission_probability, unseen_observations)
            tag_output = viterbi(observations, tags, start_probability, transition_probability, emission_probability)
            output = []
            idx = 0
            for obs in observations:
                output_token = obs + "/" + tag_output[idx]
                output.append(output_token)
                idx += 1
            output_lines.append(output)
    return output_lines


# Writes the output to file
def write_output(output, filePath):
    f = open(filePath, 'w')
    count = len(output)
    for line in output:
        str_line = ""
        for token in line:
            str_line += token
            str_line += " "
        count -= 1
        if count > 0:
            f.write(str_line.rstrip() + "\n")
        else:
            f.write(str_line.rstrip())
    f.close()


# Loads model from the model file
def load_model(file_path):
    with open(file_path) as modelFile:
        model = json.load(modelFile)
    model = ast.literal_eval(json.dumps(model, ensure_ascii=False, encoding='utf8'))
    return model

# main execution
if __name__ == '__main__':

    # loading model from hmmmodel.txt
    model = load_model("hmmmodel.txt")

    start_probability = model["start_probability"]
    transition_probability = model["transition_probability"]
    emission_probability = model["emission_probability"]

    MIN_LOG_VALUE = math.log(0.0000000000000001)

    # test file
    test_file = sys.argv[1]
    output = execute_viterbi(start_probability, transition_probability, emission_probability, test_file)

    # writing the output to file
    write_output(output, "hmmoutput.txt")