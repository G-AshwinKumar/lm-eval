import datasets as datasets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["text"] = doc["prompts"]
      doc["category"] = doc["category"]
      # doc["answer"] = "Answer here"
      return doc

    return dataset.map(_helper)  # returns back a datasets.Dataset object

def doc_to_text(doc) -> str:
    return "{}".format(
        doc["text"]
    )


def process_results_gen(doc, results):
    analyzer = SentimentIntensityAnalyzer()

    print(doc)
    print(results)
    vs = analyzer.polarity_scores(results)
    print("{}{:-<65} {}".format(doc["category"],results, str(vs)))

    return {
        "vaderSentiment": vs,
    }
