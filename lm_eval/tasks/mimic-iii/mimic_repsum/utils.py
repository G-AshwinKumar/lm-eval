#from rouge import Rouge
import sacrebleu
from collections.abc import Iterable

from rouge_score import rouge_scorer, scoring

def doc_to_text(doc) -> str:
    """
    instructions = "Instructions: You are a helpful radiology assistant. The following are questions about radiology " \
                   "reports. Summarize the findings in the report into diagnostic statements. \n" \
                   "Given the findings: {}. Q: Summarize the findings.".format(doc["text"])
    """
    return doc["extractive_notes_summ"]

def doc_to_target(doc) -> str:
    return doc["target_text"]

def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


def process_results_gen(doc, results):

    rouge_scores = rouge([doc["target_text"]], [results[0]])

    bleu_scores = bleu([doc["target_text"]], [results[0]])

    return {
        "bleu": bleu_scores,
        "rougeL": rouge_scores["rougeLsum"],
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
    }


def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    if len(refs) == 0 or len(preds) == 0:
        return 0

    refs, preds = _sacreformat(refs, preds)

    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    output:
        rouge1: Overlap of unigrams (single words) between the predicted and reference text.
        rouge2: Overlap of bigrams (pairs of consecutive words).
        rougeLsum: Measures the longest common subsequence between the predicted and reference text. It considers the recall of overlapping word sequences.
    """    
    
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    if len(refs) == 0 and len(preds) == 0:
        return {type: 1 for type in rouge_types}
    elif len(refs) == 0 and len(preds) != 0:
        return {type: 0 for type in rouge_types}
    elif len(refs) !=0 and len(preds) == 0:
        return {type: 0 for type in rouge_types}

    scorer = rouge_scorer.RougeScorer(rouge_types)

    # Add newlines between sentences to correctly compute `rougeLsum`.
    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure for type in rouge_types}