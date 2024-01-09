# Fine tuning a NLP transformer

## Description

I meanly use `transformer` library from `hugging face`. Some of these
transformer have limited language support. Typically, they support English, French and Italian.
But we would like to support also other languages. For that we need 3 datasets (train, validation and test).

## Required libraries

```sh
!pip install --upgrade pip
!pip install sentencepiece
!pip install datasets
!pip install transformers
!pip install accelerate -U
```

## Dataset definition

- test_labels.txt
- test_text.txt
- train_labels.txt
- train_text.txt
- val_labels.txt
- val_text.txt

### Definition example

#### text

```text
نوال الزغبي (الشاب خالد ليس عالمي) هههههههه أتفرجي على ها الفيديو يا مبتدئة http vía @user
تقول نوال الزغبي : http
نوال الزغبي لطيفه الفنانه الوحيده اللي كل الفيديو كليبات تبعها ماتسبب تلوث بصري ولا سمعي لو صوتها اقل من عادي
```

#### labels

```text
0
1
2
```

## Hyperparameter definition

```python
LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 32
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment" # use this to finetune the language model
MAX_LENGTH = 514
```

- LR: how much the model should learn from the new training process
- EPOCHS: how many epochs should the model be trained
- BATCH_SIZE: number of examples per each batch
- MODEL: transformer model from hugging face or from local source
- MAX_LENGTH: maximum length per sentence

## Data

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True, max_length=MAX_LENGTH)

train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])
```

### Steps

- Define the tokenizer
- Define the encondings
- Define the datasets

## Fine-tuning

```python
training_args = TrainingArguments(
    output_dir='./results',                   # output directory
    num_train_epochs=EPOCHS,                  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
    warmup_steps=100,                         # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # strength of weight decay
    logging_dir='./logs',                     # directory for storing logs
    logging_steps=10,                         # when to print log
    load_best_model_at_end=True,              # load or not best model at the end
    evaluation_strategy="steps"
)

num_labels = len(set(dataset_dict["train"]["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

trainer = Trainer(
    model=model,                              # the instantiated 🤗 Transformers model to be trained
    args=training_args,                       # training arguments, defined above
    train_dataset=train_dataset,              # training dataset
    eval_dataset=val_dataset                  # evaluation dataset
)

trainer.train()

trainer.save_model("./results/best_model") # save best model
```

### Steps

- Define the training and evaluation arguments
- Create the model
- Instantiate a trainer and train the model
- Save the definition of the best model

## Evaluate the performance of the model

```python
import np
from sklearn.metrics import classification_report

test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_raw, axis=-1)
print(classification_report(test_labels, test_preds, digits=3))
```
