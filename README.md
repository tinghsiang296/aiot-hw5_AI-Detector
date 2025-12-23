# 🤖 AI vs Human Text Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/models)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

這是一個基於深度學習的文本分類工具，旨在辨別一段文字是由 **人類撰寫** 還是由 **AI（如 ChatGPT）** 生成。本專案使用 Hugging Face 的 RoBERTa 預訓練模型，並透過 Streamlit 提供直觀的網頁互動介面。

---

## ✨ 核心功能 (Features)

* **即時偵測 (Real-time Detection):** 輸入文本後立即獲得 AI 與人類撰寫的機率百分比。
* **數據可視化 (Data Visualization):** 使用 Plotly 繪製置信度長條圖，分析結果一目了然。
* **高效能模型 (High Performance):** 採用 `chatgpt-detector-roberta` 模型，針對 GPT 生成內容有極佳的辨識度。
* **簡潔介面 (Clean UI):** 基於 Streamlit 開發，使用者體驗流暢，無需複雜設定。

---

## 🛠 技術棧 (Tech Stack)

* **程式語言:** Python
* **前端介面:** [Streamlit](https://streamlit.io/)
* **深度學習框架:** [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)
* **數據可視化:** [Plotly](https://plotly.com/python/)
* **預訓練模型:** RoBERTa



