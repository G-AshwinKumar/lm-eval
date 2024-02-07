import datasets as datasets


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["query"] = doc["question"]
      doc["choices"] = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
      doc["gold"] = doc["cop"]
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

def doc_to_text(doc) -> str:
    return "Question: {}\nChoices: {}\nAnswer:".format(
       doc["question"], doc["choices"], doc["choices"][doc["cop"]]
    )

def doc_to_choice(doc) -> str:
    return [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]

def doc_to_target(doc) -> str:
    return doc["choices"][doc["cop"]]
