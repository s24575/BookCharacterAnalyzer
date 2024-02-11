from pprint import pprint
import pandas as pd
from datasets import load_dataset
from evaluate import evaluator
from transformers import pipeline


# Compare models using conll2003 dataset

models = [
    "dslim/bert-base-NER",
    # "dbmdz/bert-large-cased-finetuned-conll03-english",
    # "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
    # "Babelscape/wikineural-multilingual-ner",
    # "Davlan/bert-base-multilingual-cased-ner-hrl"
]

data = load_dataset("conll2003", split="validation").shuffle().select(range(1000))
task_evaluator = evaluator("token-classification")

results = []
for model in models:
    print(f"Evaluating model: {model}")
    results.append(
        task_evaluator.compute(
            model_or_pipeline=model, data=data, metric="seqeval"
            )
        )

df = pd.DataFrame(results, index=models)
print(df[["overall_f1", "overall_accuracy", "overall_recall", "overall_precision", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]])
df.to_csv("evaluation_results.csv", index_label="Model")
