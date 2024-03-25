import datasets as datasets


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["query"] = doc["question"]
      doc["choices"] = doc["choices"]
      doc["gold"] = doc["choices"].index(doc["answer"][0])
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

def doc_to_text(doc) -> str:
    return "Question: {}\nOptions: {}\nAnswer:".format(
        doc["question"], doc["choices"]
    )

def doc_to_choice(doc) -> str:
    return doc["choices"]

def doc_to_target(doc) -> str:
    return doc["answer"][0]
