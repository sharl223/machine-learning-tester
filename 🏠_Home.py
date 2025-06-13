import streamlit as st
import streamlit.components.v1 as components

# ページ設定
st.set_page_config(
    page_title="ようこそ！未来予測アプリへ",
    page_icon="✨",
    layout="wide"
)

# --------------------------------------------------------------------------------
# Google Analyticsトラッキングコード
# --------------------------------------------------------------------------------
# あなたの測定ID: G-FCTKX7G62M
GA_MEASUREMENT_ID = "G-FCTKX7G62M" 

# Google Analyticsのトラッキングコード（gtag.js）
tracking_script = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());

  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
"""
# HTMLとしてスクリプトを埋め込む
# height=0にすることで、コンポーネントの表示領域をなくす
components.html(tracking_script, height=0)
# --------------------------------------------------------------------------------



# --- デザインカスタマイズ (改行修正版) ---
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

/* ★★★ ここからが今回の修正箇所 ★★★ */

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
st.title("さあ、データで未来を予測しよう！ ✨")

st.markdown("---")

st.markdown("""
### このサイトは、むずかしいプログラミングをしなくても、
### あなたのデータからAI予測モデルを作れる遊び場です。

「このお客さんは、商品を買ってくれるかな？」  
「明日の売上は、どれくらいになるんだろう？」  
「この患者さんは、検査結果が陽性？それとも陰性？」

そんな「もしも」の答えを、データと一緒に探してみませんか？
""")

st.markdown("---")

st.header("できること")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🚀 LightGBM Playground")
        st.markdown("""
        **あなたのデータで、予測モデルを育てよう！**

        お手持ちのCSVファイルをアップロードするだけで、AIの学習がスタートします。
        
        サイドバーのつまみを動かして、AIの「作戦（ハイパーパラメータ）」を調整し、最高のスコアを目指しましょう！まるでゲームのように、試行錯誤する過程を楽しめます。

        現在は、高速で性能の良い「**LightGBM**」というモデルが使えますが、将来的には他の強力なモデル（XGBoostなど）も選べるようにアップデート予定です！
        """)

with col2:
    with st.container(border=True):
        st.subheader("📚 サンプルデータ置き場")
        st.markdown("""
        **手ぶらで来ても大丈夫！**

        「試してみたいけど、ちょうどいいデータがない…」という方のために、練習用のデータセットを用意しました。
        
        有名な「タイタニック号の生存予測」や「カリフォルニアの住宅価格予測」のデータで、気軽にAIの性能を試してみてください。
        """)

st.info("まずはサイドバーの `LightGBM Playground` を選んで、サンプルデータから試してみるのがおすすめです！", icon="💡")

st.write("---")
st.header("データの準備")

with st.expander("📖 アップロードするCSVファイルの形式について", expanded=False):
    st.markdown("""
    このアプリで最高の性能を引き出すために、アップロードするCSVファイルは以下の2つのルールに従って作成してください。

    #### 1. ヘッダー行（列名）が必要です
    ファイルの1行目は、各列のデータが何を表すかを示す名前（`年齢`, `価格`, `顧客ID`など）である必要があります。
    
    #### 2. 最初の列をインデックスにしてください
    データの最初の列は、各行を一位に識別するための**インデックス（IDなど）**として扱われます。
    もしインデックスが必要ない単純なデータの場合でも、0, 1, 2... といった連番の列を最初の列としてご用意ください。

    ---
    
    #### ✔️ 正しいフォーマットの例
    以下は `titanic_cleaned.csv` の例です。`PassengerId` がインデックスとして最初の列に配置されています。
    ```csv
    PassengerId,Survived,Pclass,Sex,Age,SibSp,Parch,Fare
    1,0,3,male,22.0,1,0,7.25
    2,1,1,female,38.0,1,0,71.2883
    3,1,3,female,26.0,0,0,7.925
    ...
    ```
    
    #### ❌ 間違ったフォーマットの例
    
    **ヘッダー行がない**
    ```csv
    1,0,3,male,22.0,1,0,7.25
    2,1,1,female,38.0,1,0,71.2883
    ...
    ```
    
    **インデックス列が先頭にない**
    ```csv
    Survived,Pclass,Sex,Age,SibSp,Parch,Fare,PassengerId
    0,3,male,22.0,1,0,7.25,1
    1,1,female,38.0,1,0,71.2883,2
    ...
    ```
    """)

st.info("まずはサイドバーの `LightGBM Playground` を選んで、サンプルデータから試してみるのがおすすめです！", icon="💡")

st.write("---")
st.header("謝辞・使用技術")
st.markdown("""
このWebアプリケーションは、以下の素晴らしいオープンソースソフトウェアやデータセットを利用して構築されています。
すべての開発者とコミュニティに心より感謝申し上げます。

- **アプリケーションフレームワーク:** Streamlit
- **機械学習モデル:** LightGBM
- **データ処理・分析:** pandas, scikit-learn
- **グラフ描画:** Matplotlib, japanize-matplotlib
- **データプロファイリング:** ydata-profiling
- **サンプルデータセット:**
    - **タイタニック号の乗客データ:** Kaggle / seaborn
    - **カリフォルニアの住宅価格データ:** scikit-learn

""")
st.sidebar.success("メニューを選んで、さっそく始めよう！")