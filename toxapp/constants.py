AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

PAD_ID = 0
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
ID_TO_AA = {i: aa for aa, i in AA_TO_ID.items()}
VOCAB_SIZE = len(AA_TO_ID) + 1

