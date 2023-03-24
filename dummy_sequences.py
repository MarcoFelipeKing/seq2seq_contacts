import numpy as np

# Set the random seed
np.random.seed(42)

# Define the sequence of actions
actions = [1, 2, 3, 4, 5, 0]

# Define the number of nurses
num_nurses = 10

# Define the length of the sequence
seq_length = 20

# Create a list to hold the sequences
sequences = []

# Loop through each nurse
for i in range(num_nurses):
    # Generate a random sequence of actions
    seq = np.random.choice(actions, size=seq_length-2, replace=True)
    
    # Add the start and end actions
    seq = np.concatenate(([1], seq, [0]))
    
    # Add the sequence to the list
    sequences.append(seq)
    
# Print the sequences
for i, seq in enumerate(sequences):
    print(f"Sequence {i+1}: {seq}")