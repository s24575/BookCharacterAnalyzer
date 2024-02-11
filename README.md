# Book Character Analyzer

This website application is based around the concept of Natural Entity Recognition in fantasy books. It allows you to search for all the important features inside the book, such as characters, locations, and more. You can find a part of The Lord of the Rings book as an example in the `books/` folder.

### Setup

```
git clone https://github.com/s24575/BookCharacterAnalyzer.git
```

```
cd BookCharacterAnalyzer
```

```
python -m venv venv
```

```
.\venv\Scripts\activate # Windows
source venv/bin/activate # macOS/Linux
```

```
pip install -r requirements.txt
```

### Run the website

    python app.py

##### Go to [http://localhost:5000](http://localhost:5000)

### Evaluate models

    python ner_model_comp.py
