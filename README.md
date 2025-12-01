# 🧠 Simple BERT Fine-Tuning for Question Answering

A minimal TensorFlow implementation of **BERT fine-tuning** for the **SQuAD Question Answering** task — no sliding window, no extra preprocessing, just clean and simple code for educational and experimental use.

---
## 🤗 datasets

**SQuAD v1.1 dataset** from Datasets Library.

## 🚀 Features

- ✅ Uses **Hugging Face Transformers** (`TFBertForQuestionAnswering`)
- ✅ Token alignment via offset mapping
- ✅ Trains and evaluates on the **SQuAD v1.1 dataset**
- ✅ Includes an easy-to-use **inference function**
- ✅ Compact, well-commented, and ideal for learning

---

## 🧩 Requirements

Install the dependencies:

pip install tensorflow transformers datasets

## 🧱 Project Structure
bert-qa-finetuning/
│
├── bert_qa_train.py       # Main training script
├── README.md             
└── requirements.txt       # dependencies list

## 📈 Notes

The implementation is non-sliding, meaning it may truncate long contexts.

For production or SQuAD benchmarks, use the Hugging Face Trainer API.

This script is ideal for educational purposes, demonstrations, or lightweight fine-tuning experiments.

## 💡 Author

Created by: Krupa Jeevan
Inspired by: Hugging Face team & TensorFlow community
License: MIT License

⭐ If you find this project helpful, give it a star on GitHub!

Feel free to modify the Repo and the Code😊😊😋😋
