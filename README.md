🤖 AI vs Human Text Classifier
這是一個基於深度學習的文本分類工具，旨在辨別一段文字是由 人類撰寫 還是由 AI（如 ChatGPT） 生成。本專案使用 Hugging Face 的 RoBERTa 預訓練模型，並透過 Streamlit 提供直觀的網頁互動介面。

A deep learning-based text classification tool designed to distinguish whether a piece of text is Human-written or AI-generated (e.g., ChatGPT). This project utilizes a RoBERTa pre-trained model from Hugging Face and provides an intuitive web interface via Streamlit.

✨ 核心功能 (Features)
即時偵測 (Real-time Detection): 輸入文本後立即獲得 AI 與人類撰寫的機率百分比。

數據可視化 (Data Visualization): 使用 Plotly 繪製置信度長條圖，分析結果一目了然。

高效能模型 (High Performance): 採用 chatgpt-detector-roberta 模型，針對 GPT 生成內容有極佳的辨識度。

簡潔介面 (Clean UI): 基於 Streamlit 開發，使用者體驗流暢。

🛠 🛠 技術棧 (Tech Stack)
Language: Python

Framework: Streamlit

Deep Learning: Transformers (Hugging Face)

Data Visualization: Plotly

Model: RoBERTa (Base)
