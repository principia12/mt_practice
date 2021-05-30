def default_tokenizer(text):
    return text.split()

def default_detokenizer(tokens):
    return ' '.join(tokens)
