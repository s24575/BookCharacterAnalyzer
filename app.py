from collections import defaultdict
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load model
# pipe = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
pipe = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
# pipe = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english", grouped_entities=True)
# pipe = pipeline("ner", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
# pipe = pipeline("token-classification", model="Davlan/bert-base-multilingual-cased-ner-hrl")

def process_text_chunk(chunk):
    return pipe(chunk)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    if request.method == 'POST':

        uploaded_file = request.files['file']

        if uploaded_file:
            chunk_size = 512
            processed_size = 0
            entities = []

            # Read the file in chunks
            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                
                chunk_text = chunk.decode('utf-8')
                entities.extend(process_text_chunk(chunk_text))
                processed_size += chunk_size
                print(processed_size)

            entity_group_full_name = {
                "PER": "Person",
                "LOC": "Location",
                "MISC": "Miscellaneous",
                "ORG": "Organization"
            }

            # Group entities by type and count frequencies
            entity_counter = defaultdict(lambda: defaultdict(int))
            for entity in entities:
                entity_group = entity_group_full_name[entity['entity_group']]
                entity_word = entity['word']
                entity_counter[entity_group][entity_word] += 1
            
            # Sort them, entities with the most occurances first
            for entity_group, inner_dict in entity_counter.items():
                entity_counter[entity_group] = dict(sorted(inner_dict.items(), key=lambda x: x[1], reverse=True))

            return render_template('result.html', entity_counter=entity_counter)

if __name__ == '__main__':
    app.run(debug=True)
