from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

# TODO: Can experiment with to reduce computation times. especially on the 'Essays' Dataset.
if __name__ == '__main__':

    # Text to summarize
    text = """"""

    summarize_to_n_sentences = 5

    # Parse the text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Initialize the summarizer
    summarizer = LexRankSummarizer()
    # Summarize the text
    # Number of sentences in the summary
    summary = summarizer(parser.document, summarize_to_n_sentences)

    # Print the summary
    for sentence in summary:
        print(sentence)