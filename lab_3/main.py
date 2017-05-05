from dataset import ConversationPreprocessor
from rnn import *

proc = ConversationPreprocessor('selected_conversations.txt', 2, 30)
proc.preprocess()
proc.create_minibatches()

run_language_model(proc, 5)
