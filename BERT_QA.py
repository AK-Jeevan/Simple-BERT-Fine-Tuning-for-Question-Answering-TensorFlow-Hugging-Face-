# This script fine-tunes a pre-trained BERT model to answer questions by predicting answer spans in text.

import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForQuestionAnswering
from datasets import load_dataset

MODEL = "bert-base-uncased"
MAX_LEN = 384
BATCH = 8
EPOCHS = 2
LR = 2e-5

# 1) load data & tokenizer
squad = load_dataset("squad")
tokenizer = BertTokenizerFast.from_pretrained(MODEL)

# 2) prepare dataset: token-level start/end using offsets
def prepare(example):
    enc = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=MAX_LEN,
        padding="max_length",
        return_offsets_mapping=True
    )
    offsets = enc.pop("offset_mapping")

    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])

    # find token indices that cover the answer (within the single window)
    token_start, token_end = 0, len(enc["input_ids"]) - 1
    # find first token of context (sequence_ids == 1)
    seq_ids = enc.sequence_ids()
    while token_start < len(seq_ids) and seq_ids[token_start] != 1:
        token_start += 1
    while token_end >= 0 and seq_ids[token_end] != 1:
        token_end -= 1

    # default to CLS if not inside window
    cls_index = enc["input_ids"].index(tokenizer.cls_token_id)
    if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
        enc["start_positions"] = cls_index
        enc["end_positions"] = cls_index
    else:
        # find exact token indices
        i = token_start
        while i <= token_end and offsets[i][0] <= start_char:
            i += 1
        enc["start_positions"] = i - 1
        j = token_end
        while j >= token_start and offsets[j][1] >= end_char:
            j -= 1
        enc["end_positions"] = j + 1

    return enc

# map (this is a simple non-batched mapping; you can batched-map later)
train = squad["train"].map(prepare)
val   = squad["validation"].map(prepare)

# set TF format and make datasets
train.set_format(type="tensorflow", columns=["input_ids","attention_mask","token_type_ids","start_positions","end_positions"])
val.set_format(type="tensorflow", columns=["input_ids","attention_mask","token_type_ids","start_positions","end_positions"])

def to_tf(ds):
    features = {k: tf.constant(ds[k]) for k in ["input_ids","attention_mask","token_type_ids"]}
    labels = {"start_positions": tf.constant(ds["start_positions"]), "end_positions": tf.constant(ds["end_positions"])}
    tfds = tf.data.Dataset.from_tensor_slices((features, labels))
    return tfds.shuffle(1024).batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = to_tf(train)
val_ds = to_tf(val)

# 3) model, compile, train
model = TFBertForQuestionAnswering.from_pretrained(MODEL, from_pt=True)
losses = {
    "start_positions": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "end_positions": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
}
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=losses)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 4) inference
def answer(question, context):
    enc = tokenizer(question, context, truncation="only_second", max_length=MAX_LEN, return_tensors="tf")
    out = model(enc)
    start = int(tf.argmax(out.start_logits, axis=1)[0])
    end   = int(tf.argmax(out.end_logits,   axis=1)[0])
    if start > end:
        return ""
    return tokenizer.decode(enc["input_ids"][0][start:end+1], skip_special_tokens=True)
