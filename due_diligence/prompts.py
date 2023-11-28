
def format_instruction(sample):
    return f"""Clause: "{sample['sentence']}"\nInstruction: You are an expert lawyer specializing in due diligence for Mergers & Acquisitions. Your task is to assess the provided clause in a contract. Determine whether it poses a significant liability or risk for the transaction. Respond with either 'Yes' or 'No'.\n\nAnswer:"""
def format_instruction(sample):
    return f"""Sentence: "{sample['sentence']}"\nInstruction: Your task is to identify provisions for mergers and aquisitions? Does this Sentence pose a major liability or risk for a transaction? Yes or No?\n\nAnswer:"""
    return f"""Sentence: {sample['sentence']}\nInstruction: Does this sentence pose a major liability or risk for a transaction in due diligence in mergers and aquisitions? Yes or No?\n\nAnswer:"""



def format_instruction(sample):
    return f"""Sentence: {sample['sentence']}\nInstruction: You are an expert lawyer in due diligence for Mergers and Aquisitions. Does this Sentence pose a major liability or risk for the transaction. Yes or No?\n\nAnswer:"""
    #return f"""Instruction:\nAs an expert lawyer specializing in M&A due diligence, please evaluate the following clause: "{sample['sentence']}." Does it present a significant liability or risk for the transaction? Yes or No?\n\nAnswer:"""
def format_instruction(sample):
    return f"""Sentence: {sample['sentence']}\nInstruction: You are an expert lawyer for due diligence in mergers and aquisitions. Your task is to identify provisions. Does this sentence pose a major liability or risk for the transaction? yes or no?\n\nAnswer:"""
    return f"""Sentence: {sample['sentence']}\nInstruction: You are an expert lawyer for due diligence in mergers and aquisitions. Does this sentence pose a major liability or risk for the transaction? Yes or No?\n\nAnswer:"""


def format_instruction(sample):
	return f"""### Instruction:
You are an expert lawyer for due diligence in mergers and aquisitions. Does the Input pose a major liability or risk for a transaction?

### Replay:
{sample['sentence']}

### Response:
"""



def format_instruction(sample):
    return f"""### Instruction:
In due diligence tasks, lawyers seek to find a set of standard passage-types called provisions, that typically correspond to an increased risk in a transaction. These range from what happens to a contract when one party is acquired (“change of control”), to what parties are involved in the contracts
to what happens when a contract is terminated early. You are an expert lawyer specializing in due diligence in mergers and acquisitions. Does the following Sentence present a significant liability or risk? Reply with 'Yes ' or 'No'.

### Sentence:
{sample['sentence']}

### Answer:
"""


def format_instruction(sample):
    return f"""### Instruction:
Does the following sentence present a liability or risk for a mergers and aquisitions transaction? Reply with 'Yes ' or 'No'.

### Sentence:
{sample['sentence']}

### Answer:
"""

