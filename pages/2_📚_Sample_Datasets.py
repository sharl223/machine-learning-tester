import streamlit as st
import pandas as pd
from io import BytesIO

# --- ページ設定 ---
st.set_page_config(page_title="サンプルデータ置き場", page_icon="📚")

# --- デザインカスタマイズ ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+Rounded+1c&display=swap');

/* --- 動くグラデーション背景 --- */
@keyframes gradient-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
[data-testid="stAppViewContainer"] > .main {
    background-image: linear-gradient(-45deg, #051937, #004d7a, #008793, #00bf72, #a8eb12);
    background-size: 400% 400%;
    animation: gradient-animation 15s ease infinite;
}

/* --- グラスモーフィズム (サイドバーとコンテナ) --- */
[data-testid="stSidebar"] > div:first-child,
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 16px;
}
[data-testid="stVerticalBlockBorderWrapper"] { padding: 24px; }

/* --- ボタンのホバーエフェクト --- */
button[data-testid="stButton"] > button {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
button[data-testid="stButton"] > button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* --- 全体のフォント --- */
html, body, [class*="st-"], [class*="css-"] { font-family: 'M PLUS Rounded 1c', sans-serif; }

/* --- ヘッダーの文字色と改行を調整 --- */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF;
    word-break: keep-all !important; /* 単語の途中での改行を防ぐ(復活・強化) */
}

/* --- サイドバーの文字色 --- */
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

/* --- 詳細スコアのメトリック表示 --- */
[data-testid="stMetric"] { text-align: center; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] p { font-size: 0.9rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.75rem !important; }

</style>
""", unsafe_allow_html=True)


# --- メインコンテンツ ---
st.title("📚 サンプルデータ置き場")

st.info("""
AIモデルの学習と性能テストを体験できるよう、2種類のデータセットを用意しました。
以下の手順で試してみてください。

1.  **学習用データ**をダウンロードし、`LightGBM Playground` でアップロードしてモデルを**学習**させます。
2.  **予測用データ**をダウンロードし、学習後に出現する「未来を予測してみよう！」セクションでアップロードします。
3.  予測結果と、実際の答えを比べた**正解率**が表示されることを確認します。
""", icon="💡")
st.write("---")

# --- タイタニック ---
st.subheader("🚢 タイタニック号の生存者を探せ！ (分類問題)")
st.write("""
乗客のプロフィールから生存者を予測する分類問題。まずは**学習用データ**でAIを鍛え、次に**予測用データ**で未知の乗客の運命を予測し、その正解率を確かめてみましょう。
""")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("##### 1. 学習用データ")
        try:
            with open("titanic_train.csv", "rb") as file:
                st.download_button(
                    label="titanic_train.csv をダウンロード",
                    data=file,
                    file_name="titanic_train.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            df_train = pd.read_csv("titanic_train.csv", index_col=0)
            st.dataframe(df_train.head(), height=220, use_container_width=True)
        except FileNotFoundError:
            st.error("`titanic_train.csv` が見つかりません。")

with col2:
    with st.container(border=True):
        st.markdown("##### 2. 予測用データ")
        try:
            with open("titanic_predict.csv", "rb") as file:
                st.download_button(
                    label="titanic_predict.csv をダウンロード",
                    data=file,
                    file_name="titanic_predict.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            df_predict = pd.read_csv("titanic_predict.csv", index_col=0)
            st.dataframe(df_predict.head(), height=220, use_container_width=True)
        except FileNotFoundError:
            st.error("`titanic_predict.csv` が見つかりません。")

st.write("---")

# --- カリフォルニア ---
st.subheader("🏡 カリフォルニアの住宅価格を当てろ！ (回帰問題)")
st.write("""
地域の情報から住宅価格を予測する回帰問題。**学習用データ**で価格予測モデルを作り、**予測用データ**でどれだけ実際の価格に近い値を予測できるか（決定係数 R2）を見てみましょう。
""")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("##### 1. 学習用データ")
        try:
            with open("california_train.csv", "rb") as file:
                st.download_button(
                    label="california_train.csv をダウンロード",
                    data=file,
                    file_name="california_train.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            df_train = pd.read_csv("california_train.csv", index_col=0)
            st.dataframe(df_train.head(), height=220, use_container_width=True)
        except FileNotFoundError:
            st.error("`california_train.csv` が見つかりません。")

with col2:
    with st.container(border=True):
        st.markdown("##### 2. 予測用データ")
        try:
            with open("california_predict.csv", "rb") as file:
                st.download_button(
                    label="california_predict.csv をダウンロード",
                    data=file,
                    file_name="california_predict.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            df_predict = pd.read_csv("california_predict.csv", index_col=0)
            st.dataframe(df_predict.head(), height=220, use_container_width=True)
        except FileNotFoundError:
            st.error("`california_predict.csv` が見つかりません。")