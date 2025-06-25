# 🧠 Multilingual Next-Word Prediction using GPT-2 (Bhashathon 2025 Winner 🏆)

This project was developed for **Bhashathon 2025**, focusing on building a robust **multilingual next-word prediction model** for six Indian languages using a custom-trained GPT-2 architecture. Our model was trained from scratch, achieving competitive results on limited hardware while tackling the challenges of low-resource Indic languages.

## 🧑‍🤝‍🧑 Team: ThinkingRock

## 🧩 Problem Statement

**Task:** Build a multilingual language model for **next-word prediction** in:

* Hindi
* Marathi
* Odia
* Malayalam
* Kannada
* Gujarati

---

## 📦 Dataset

* **Total Size:** 73 GB
* **Token Count:** 3.2 Billion tokens after preprocessing
* **Sources:**

  * Organizers’ dataset (for all languages)
  * [IndicCorp v2](https://github.com/AI4Bharat/IndicCorp) (for all except Odia)

---

## 🧼 Preprocessing Pipeline

* **Parallelized Preprocessing** using `ProcessPoolExecutor`
* **Chunked Reading** (10,000 lines per batch)
* **Data Cleaning:** HTML tag removal, Unicode normalization, language-specific corrections
* **Deduplication** using MD5 hashing
* **Script Verification** (≥60% valid script)
* **Sentence Segmentation** using punctuation (e.g. `।`, `.`, `?`, `!`)
* **Reservoir Sampling** for memory-efficient subset generation

---

## 🔤 Tokenization

* Trained **SentencePiece** tokenizer using **Unigram model**
* **Vocab Size:** 50,304
* **Character Coverage:** 99.95%
* **Languages:** Hindi, Gujarati, Kannada, Marathi, Malayalam, Odia
* Balanced sampling from all languages (10M sentences total)

---

## 🧠 Model Architecture

Custom GPT-2 implementation in **PyTorch**:

* **Model Size:** 124M parameters
* **Layers:** 12
* **Attention Heads:** 12
* **Embedding Size:** 768
* **Dropout:** 0.0
* **Block Size:** 1024
* **Training Time:** \~24 hours
* **Hardware:** NVIDIA A10G GPU (24GB), 4 vCPUs, 16GB RAM

### ✅ Optimizations:

* Pre-Layer Normalization
* Flash Attention
* Causal Masking Optimizations
* Unigram tokenization (for better subword capture in Indic languages)

---

## 🏋️ Training Strategy

* **Batch Size:** 16 (train), 24 (test)
* **Effective Token Batch:** 500K tokens per forward pass via **gradient accumulation**
* **Optimizer:** AdamW with cosine decay
* **Total Tokens Trained:** 4.1 Billion
* **Learning Rate:** 6e-4 → 6e-5 (cosine decay)

---

## 📈 Evaluation

* **Metric:** Cross-Entropy Loss & Perplexity
* **Logging:** Weights & Biases (W\&B)
* **Inference Speed:** 56K tokens/sec

---

## ❌ Known Challenges

* Loss plateaued after 7000 steps
* Low-resource language (Odia) underperformance
* Coherence drops in long-form completions
* Grammatical inconsistencies in low-data settings

---

## 🔮 Future Directions

* Fine-tune pre-trained multilingual models for better initialization
* Data augmentation for underrepresented languages
* Larger model scaling and smarter dynamic sampling strategies
* Adaptive learning rate tuning to avoid plateaus

## 🏁 Run Instructions


read [replicate_results.md](replicate_results.md) for : 
1. Train from scratch
2. Inference
3. Tokenizer Training


---

## 📜 License

MIT License. See `LICENSE` file for more details.
