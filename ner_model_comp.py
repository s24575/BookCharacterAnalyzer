from typing import List, Tuple
from transformers import AutoTokenizer
import pandas as pd 
from transformers import pipeline
from sklearn.metrics import classification_report


def create_iob_tags(labels: List[Tuple[int, int, str]], wordpieces: Tuple[int, int]):
    tags = []

    # Assign IOB tags based on the provided labels
    for token_start, token_end in wordpieces:
        token_label = 'O'
        for start, end, label in labels:
            if start == token_start:
                token_label = f'B-{label}'
                break
            elif start < token_start and token_end <= end:
                token_label = f'I-{label}'
                break
        tags.append(token_label)

    return tags


def parse_pipeline_output(model_pred, wordpieces: Tuple[int, int]):
    tags = []
    for token_start, token_end in wordpieces:
        tag = 'O'
        for pred in model_pred:
            if token_start == pred['start'] and token_end == pred['end']:
                tag = pred['entity']
                break
        tags.append(tag)
    return tags


def main():
    # Compare models using custom labeled dataset

    models = [
        "dslim/bert-base-NER",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "Babelscape/wikineural-multilingual-ner",
        "Davlan/bert-base-multilingual-cased-ner-hrl"
    ]

    dataset = pd.read_json(path_or_buf='test_dataset.jsonl', lines=True)

    for model_name in models:
        print(f"Evalutate {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("ner", model=model_name, tokenizer=tokenizer)

        true_labels_list = []
        pred_labels_list = []

        for _, row in dataset.iterrows():
            text = row['text']
            true_word_labels = row['label']
            wordpieces = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']

            true_labels = create_iob_tags(true_word_labels, wordpieces)

            pred_output = pipe(text)
            pred_labels = parse_pipeline_output(pred_output, wordpieces)


            true_labels_list.extend(true_labels)
            pred_labels_list.extend(pred_labels)
        
        # Evaluate
        report = classification_report(true_labels_list, pred_labels_list, output_dict=True)
        df = pd.DataFrame(report).transpose()

        # Save the results to CSV
        csv_filename = f"results/{model_name.replace('/', '_')}_results.csv"
        df.to_csv(csv_filename)

        # Save the results to a text file
        txt_filename = f"results/{model_name.replace('/', '_')}_results.txt"
        with open(txt_filename, 'w') as txt_file:
            txt_file.write(df.to_string())


if __name__ == '__main__':
    main()