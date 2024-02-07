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
    instruction = "You are a medical doctor answering real-world medical entrance exam questions. Based " \
                  "on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying " \
                  "health, disease, patient care, and modes of therapy, answer the following multiple-choice question. " \
                  "Select one correct answer from A to D. Base your answer on the current and standard practices " \
                  "referenced in medical guidelines."
    return "Instruction: {}\nQuestion: {}\nChoices: {}\nAnswer:".format(
        instruction, doc["question"], doc["choices"], doc["choices"][doc["cop"]]
    )

def doc_to_choice(doc) -> str:
    return [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]

def doc_to_target(doc) -> str:
    return doc["choices"][doc["cop"]]
