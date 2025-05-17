# %% [markdown]
# ## Homework 3: Symbolic Music Generation Using Markov Chains

# %% [markdown]
# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# %%
# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

# %%
# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)

# %% [markdown]
# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509).
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# %%
midi_files = glob('PDMX_subset/*.mid')
len(midi_files)

# %% [markdown]
# ### Train a tokenizer with the REMI method in MidiTok

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

# %% [markdown]
# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# %%
# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens

# %% [markdown]
# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# %%
def note_extraction(midi_file):
    # Q1a: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    pitches = []
    for token in tokens:
        if token.startswith('Pitch_'):
            pitch = int(token.split('_')[1])
            pitches.append(pitch)

    return pitches

# %%
def note_frequency(midi_files):
    # Q1b: Your code goes here
    note_freq = defaultdict(int)

    for midi_file in midi_files:
        pitches = note_extraction(midi_file)
        for pitch in pitches:
            note_freq[pitch] += 1

    return note_freq


# %% [markdown]
# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# %%
note_counts = note_frequency(midi_files)
total = np.sum(list(note_counts.values()))
for i in note_counts:
    print(note_counts[i])

# %%
def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}
    total = np.sum(list(note_counts.values()))

    # Q2: Your code goes here
    for i in note_counts:
        unigramProbabilities[i] = note_counts[i] / total

    return unigramProbabilities

# %% [markdown]
# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# %%
def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    # Q3a: Your code goes here
    for midi_file in midi_files:
        pitches = note_extraction(midi_file)
        for i in range(len(pitches) - 1):
            prev_note = pitches[i]
            next_note = pitches[i+1]
            bigramTransitions[prev_note].append(next_note)

    for prev_note, next_notes in bigramTransitions.items():
        counts = defaultdict(int)
        for note in next_notes:
            counts[note] += 1
        unique_counts = list(counts.keys())
        total = sum(counts.values())
        bigramTransitions[prev_note] = unique_counts
        bigramTransitionProbabilities[prev_note] = [counts[n] / total for n in unique_counts]
        
    
    return bigramTransitions, bigramTransitionProbabilities

# %%
note_bigram_probability(midi_files)

# %%
def sample_next_note(note):
    # Q3b: Your code goes here
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    probs = bigramTransitionProbabilities[note]
    notes = bigramTransitions[note]
    return np.random.choice(notes, p=probs)

# %% [markdown]
# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)

    # Q4: Your code goes here
    # Can use regular numpy.log (i.e., natural logarithm)
    notes = note_extraction(midi_file)
    N = len(notes)
    log_probs = []

    for i in range(N):
        curr_note = notes[i]
        if i == 0:
            prob = unigramProbabilities.get(curr_note, 0.000000001)
        else:
            prev_note = notes[i-1]
            if prev_note in bigramTransitionProbabilities:
                next_notes = bigramTransitions[prev_note]
                probs = bigramTransitionProbabilities[prev_note]
                if curr_note in next_notes:
                    index = next_notes.index(curr_note)
                    prob = probs[index]
                else:
                    print("curr not is not in next notes, using small prob instead")
                    prob = 0.000000001
            else:
                print("prev_note not in bigramTransitionProbabilities")
                prob = 0.000000001
        log_probs.append(np.log(prob))

    score = np.exp(-1/N * np.sum(log_probs))
    return score

# %% [markdown]
# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file.
# 
#     The perplexity of this model is defined as
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)

    # Q5a: Your code goes here
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(2, len(notes)):
            key = (notes[i-2], notes[i-1])
            next_note = notes[i]
            trigramTransitions[key].append(next_note)

    trigramTransitionProbabilities = {}

    for key, next_notes in trigramTransitions.items():
        counts = defaultdict(int)
        for note in next_notes:
            counts[note] += 1
        unique_next_notes = list(counts.keys())
        total = sum(counts.values())
        probs = [counts[n] / total for n in unique_next_notes]
        trigramTransitions[key] = unique_next_notes
        trigramTransitionProbabilities[key] = probs

    return trigramTransitions, trigramTransitionProbabilities

# %%
def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)

    # Q5b: Your code goes here
    notes = note_extraction(midi_file)
    N = len(notes)

    log_probs = []

    for i in range(N):
        if i == 0:
            prob = unigramProbabilities.get(notes[i], 0.00000001)
        elif i == 1:
            prev = notes[i-1]
            if prev in bigramTransitionProbabilities:
                next_notes = bigramTransitions[prev]
                probs = bigramTransitionProbabilities[prev]
                if notes[i] in next_notes:
                    index = next_notes.index(notes[i])
                    prob = probs[index]
                else:
                    prob = 0.00000001
            else:
                prob = 0.0000001
        else: # i >= 2
            key = (notes[i - 2], notes[i - 1])
            if key in trigramTransitionProbabilities:
                next_notes = trigramTransitions[key]
                probs = trigramTransitionProbabilities[key]
                if notes[i] in next_notes:
                    idx = next_notes.index(notes[i])
                    prob = probs[idx]
                else:
                    prob = 0.00000001
            else:
                prob = 0.00000001

        log_probs.append(np.log(prob))

    perplexity = np.exp(-np.mean(log_probs))
    return perplexity

# %% [markdown]
# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# %%
duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}

# %% [markdown]
# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# %%
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens[:20]

# %%
def beat_extraction(midi_file):
    # Q6: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    beat_info = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith('Position_'):
            position = int(token.split('_')[1])

            # for duration token
            duration_token = tokens[i+3]
            duration = duration_token.split('_')[1]
            beat_length = duration2length[duration]
            
            beat_info.append((position, beat_length))
        i += 1

    return beat_info

# %% [markdown]
# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# %%
def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)

    # Q7: Your code goes here
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        beat_lengths = [length for _, length in beat_info]

        for i in range(len(beat_lengths) - 1):
            prev = beat_lengths[i]
            curr = beat_lengths[i + 1]
            bigramBeatTransitions[prev].append(curr)
    
    for prev_length, next_lengths in bigramBeatTransitions.items():
        counts = defaultdict(int)
        for length in next_lengths:
            counts[length] += 1
        unique_lengths = list(counts.keys())
        total = sum(counts.values())
        probs = [counts[l] / total for l in unique_lengths]

        bigramBeatTransitions[prev_length] = unique_lengths
        bigramBeatTransitionProbabilities[prev_length] = probs


    return bigramBeatTransitions, bigramBeatTransitionProbabilities

# %% [markdown]
# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# %%
def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)

    # Q8a: Your code goes here
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file)
        for position, length in beat_info:
            transition_counts[position][length] += 1
    
    # Convert to required format
    for position, length_counts in transition_counts.items():
        lengths = list(length_counts.keys())
        total = sum(length_counts.values())
        probs = [length_counts[length]/total for length in lengths]
        
        bigramBeatPosTransitions[position] = lengths
        bigramBeatPosTransitionProbabilities[position] = probs

    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities


# %%
def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed
    beat_length_counts = defaultdict(int)
    total_beats = 0
    
    for midi_f in midi_files:
        beat_info = beat_extraction(midi_f)
        for _, length in beat_info:
            beat_length_counts[length] += 1
            total_beats += 1
    
    beat_unigram_probs = {length: count/total_beats for length, count in beat_length_counts.items()}
    
    beat_info = beat_extraction(midi_file)
    beat_lengths = [length for _, length in beat_info]
    beat_positions = [pos for pos, _ in beat_info]

    # perplexity for Q7
    small_num = 1e-5 
    log_sum_Q7 = 0
    count_Q7 = 0
    
    for i in range(len(beat_lengths)):
        curr_length = beat_lengths[i]

        if i == 0:
            # Use unigram for the first note
            prob = beat_unigram_probs.get(curr_length, small_num)
        else:
            prev_length = beat_lengths[i - 1]
            if prev_length in bigramBeatTransitions:
                next_lengths = bigramBeatTransitions[prev_length]
                probs = bigramBeatTransitionProbabilities[prev_length]
                mapping = dict(zip(next_lengths, probs))
                prob = mapping.get(curr_length, beat_unigram_probs.get(curr_length, small_num))
            else:
                prob = beat_unigram_probs.get(curr_length, small_num)

        log_sum_Q7 += np.log(prob)
        count_Q7 += 1

    # perplexity for Q8
    log_sum_Q8 = 0
    count_Q8 = 0
    
    for i in range(len(beat_lengths)):
        pos = beat_positions[i]
        curr_length = beat_lengths[i]
        
        if pos in bigramBeatPosTransitions:
            next_lengths = bigramBeatPosTransitions[pos]
            probs = bigramBeatPosTransitionProbabilities[pos]
            mapping = dict(zip(next_lengths, probs))
            prob = mapping.get(curr_length, beat_unigram_probs.get(curr_length, small_num))
        else:
            prob = beat_unigram_probs.get(curr_length, small_num)
        
        log_sum_Q8 += np.log(prob)
        count_Q8 += 1

    perplexity_Q7 = np.exp(-log_sum_Q7 / count_Q7) if count_Q7 > 0 else float('inf')
    perplexity_Q8 = np.exp(-log_sum_Q8 / count_Q8) if count_Q8 > 0 else float('inf')

    return perplexity_Q7, perplexity_Q8

# %% [markdown]
# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity.
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)

    for midi_file in midi_files:
        beat_info = beat_extraction(midi_file) 
        for i in range(1, len(beat_info)):
            prev_length = beat_info[i - 1][1]
            curr_pos = beat_info[i][0]
            curr_length = beat_info[i][1]

            key = (prev_length, curr_pos)
            trigramBeatTransitions[key].append(curr_length)

    trigramBeatTransitionProbabilities = {}

    for key, next_lengths in trigramBeatTransitions.items():
        counts = defaultdict(int)
        for length in next_lengths:
            counts[length] += 1
        unique_lengths = sorted(counts.keys())
        total = sum(counts.values())
        probs = [counts[l] / total for l in unique_lengths]

        trigramBeatTransitions[key] = unique_lengths
        trigramBeatTransitionProbabilities[key] = probs

    return trigramBeatTransitions, trigramBeatTransitionProbabilities


# %%
def beat_trigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)

    # Compute beat unigram probabilities
    beat_length_counts = defaultdict(int)
    total_beats = 0
    for midi_f in midi_files:
        beat_info = beat_extraction(midi_f)
        for _, length in beat_info:
            beat_length_counts[length] += 1
            total_beats += 1
    beat_unigram_probs = {length: count / total_beats for length, count in beat_length_counts.items()}

    small_num = 1e-5
    beat_info = beat_extraction(midi_file)
    if len(beat_info) == 0:
        return float('inf')

    log_probs = []
    for i in range(len(beat_info)):
        curr_length = beat_info[i][1]

        if i == 0:
            prob = beat_unigram_probs.get(curr_length, small_num)
        
        elif i == 1:
            prev_length = beat_info[i - 1][1]
            if prev_length in bigramBeatTransitions:
                next_lengths = bigramBeatTransitions[prev_length]
                probs = bigramBeatTransitionProbabilities[prev_length]
                mapping = dict(zip(next_lengths, probs))
                prob = mapping.get(curr_length, beat_unigram_probs.get(curr_length, small_num))
            else:
                prob = beat_unigram_probs.get(curr_length, small_num)
        
        else:
            prev_length = beat_info[i - 1][1]
            prev_pos = beat_info[i][0]
            key = (prev_length, prev_pos)

            if key in trigramBeatTransitionProbabilities:
                next_lengths = trigramBeatTransitions[key]
                probs = trigramBeatTransitionProbabilities[key]
                mapping = dict(zip(next_lengths, probs))
                prob = mapping.get(curr_length, beat_unigram_probs.get(curr_length, small_num))
            elif prev_pos in bigramBeatPosTransitions:
                next_lengths = bigramBeatPosTransitions[prev_pos]
                probs = bigramBeatPosTransitionProbabilities[prev_pos]
                mapping = dict(zip(next_lengths, probs))
                prob = mapping.get(curr_length, beat_unigram_probs.get(curr_length, small_num))
            else:
                prob = beat_unigram_probs.get(curr_length, small_num)

        log_probs.append(np.log(prob))

    perplexity = np.exp(-np.mean(log_probs)) if len(log_probs) > 0 else float('inf')
    return perplexity


# %% [markdown]
# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# %%
def music_generate(length):
    # sample notes
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)

    # Q10: Your code goes here ...
    sampled_notes = [62, 64]
    for i in range(2, length):
        context = (sampled_notes[-2], sampled_notes[-1])
        if context in trigramTransitionProbabilities:
            next_notes = trigramTransitions[context]
            probs = trigramTransitionProbabilities[context]
            next_note = np.random.choice(next_notes, p=probs)
        else:
            next_note = np.random.choice(list(unigramProbabilities.keys()), p=list(unigramProbabilities.values()))
        sampled_notes.append(next_note)


    # sample beats
    sampled_beats = [8]
    beat_positions = [0]
    position = 0
    for i in range(1, length):
        prev_length = sampled_beats[-1]
        key = (prev_length, position)
        if key in trigramBeatTransitionProbabilities:
            next_lengths = trigramBeatTransitions[key]
            probs = trigramBeatTransitionProbabilities[key]
            beat_length = np.random.choice(next_lengths, p=probs)
        else:
            beat_length = 8
        sampled_beats.append(beat_length)

        position += beat_length
        if position >= 32:
            position = 0
        beat_positions.append(position)

    midi = MIDIFile(1)
    midi.addTempo(track=0, time=0, tempo=120)

    current_time = 0
    for pitch, beat_len in zip(sampled_notes, sampled_beats):
        duration = beat_len / 8
        midi.addNote(track=0, channel=0, pitch=pitch, time=current_time, duration=duration, volume=100)
        current_time += duration

    with open("q10.mid", "wb") as f:
        midi.writeFile(f)

    return 0


