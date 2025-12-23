import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. 頁面設定
st.set_page_config(page_title="AI vs Human 文本分類器", page_icon="🤖")

# 2. 載入模型 (使用快取避免重複載入)
@st.cache_resource
def load_classifier():
    # 使用 Hello-SimpleAI 的專用偵測模型，這對 ChatGPT 生成內容很有效
    model_path = "Hello-SimpleAI/chatgpt-detector-roberta"
    return pipeline("text-classification", model=model_path, top_k=None)

classifier = load_classifier()

# 3. UI 介面
st.title("🤖 AI vs 👤 Human 文本分類工具")
st.write("輸入一段英文文本（目前該模型對英文支援度最高），判斷其為人工撰寫或 AI 生成。")

user_input = st.text_area("請輸入待測文本：", height=200, placeholder="在此輸入文章內容...")

if st.button("開始分析"):
    if user_input.strip() == "":
        st.warning("請先輸入內容！")
    else:
        with st.spinner("分析中，請稍候..."):
            # 4. 進行預測
            # 模型輸出通常為: [{'label': 'ChatGPT', 'score': 0.99}, {'label': 'Human', 'score': 0.01}]
            results = classifier(user_input[:512])[0]  # 模型通常限制 512 tokens
            
            # 整理數據
            df_results = pd.DataFrame(results)
            
            # 轉換標籤名稱以便顯示
            label_map = {"ChatGPT": "AI 生成", "Human": "人類撰寫"}
            df_results['label'] = df_results['label'].map(label_map)
            
            # 取得具體數值
            ai_score = df_results[df_results['label'] == 'AI 生成']['score'].values[0]
            human_score = df_results[df_results['label'] == '人類撰寫']['score'].values[0]

            # 5. 顯示結果
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("AI 機率", f"{ai_score:.2%}")
            col2.metric("人類機率", f"{human_score:.2%}")

            if ai_score > human_score:
                st.error(f"判定結果：這段文字極有可能是 **AI 生成** 的。")
            else:
                st.success(f"判定結果：這段文字看起來是由 **人類撰寫** 的。")

            # 6. 可視化圖表
            st.subheader("統計量分析")
            fig = px.bar(
                df_results, 
                x='label', 
                y='score', 
                color='label',
                labels={'score': '信心程度', 'label': '分類'},
                color_discrete_map={'AI 生成': '#EF553B', '人類撰寫': '#636EFA'}
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

# 頁尾說明
st.caption("備註：本工具使用 RoBERTa 預訓練模型。AI 偵測技術並非 100% 準確，結果僅供參考。")