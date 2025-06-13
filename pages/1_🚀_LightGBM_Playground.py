import streamlit as st
import pandas as pd
import lightgbm as lgb
import optuna
import shap
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
#import japanize_matplotlib
from io import BytesIO
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

# --- ページの基本設定 ---
st.set_page_config(page_title="LightGBM Playground", page_icon="🚀", layout="wide")

# SHAPプロットをStreamlitで表示するためのヘルパー関数 (文字色修正版)
def st_shap(plot, height=None):
    custom_style = """
    <style>
        /* テキストラベルの色とフォント */
        div[id^='shap-force-plot-'] {
            color: #FFFFFF !important;
            font-family: 'M PLUS Rounded 1c', sans-serif !important;
        }
        /* 軸の「数値」の色を変更 */
        .tick text {
            fill: #FFFFFF !important;
        }
        /* 軸の「線」の色を変更 */
        .tick line, .domain {
            stroke: #FFFFFF !important;
        }
    </style>
    """
    shap_html = f"<head>{shap.getjs()}{custom_style}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- デザインカスタマイズ ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+Rounded+1c&display=swap');
[data-testid="stAppViewContainer"] > .main {
    background-image: linear-gradient(-45deg, #051937, #004d7a, #008793, #00bf72, #a8eb12);
    background-size: 400% 400%; animation: gradient-animation 15s ease infinite;
}
@keyframes gradient-animation {
    0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; }
}
[data-testid="stSidebar"] > div:first-child, [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 16px;
}
[data-testid="stVerticalBlockBorderWrapper"] { padding: 24px; }
button[data-testid="stButton"] > button {
    transition: transform 0.2s ease, box-shadow 0.2s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
button[data-testid="stButton"] > button:hover {
    transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
html, body, [class*="st-"], [class*="css-"] { font-family: 'M PLUS Rounded 1c', sans-serif; }
h1, h2, h3, h4, h5, h6 { color: #FFFFFF; word-break: keep-all !important; }
[data-testid="stSidebar"] label { color: #FFFFFF !important; }
[data-testid="stMetric"] { text-align: center; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] p { font-size: 0.9rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.75rem !important; }
</style>
""", unsafe_allow_html=True)


# --- メインコンテンツ ---
st.title("🚀 LightGBM プレイグラウンド")
st.write("ここでは、あなたのデータを使ってAI予測モデルを育て、その性能を試すことができます。")

# --- サイドバー ---
st.sidebar.header("⚙️ 設定")
st.sidebar.subheader("1. 学習用データ")
uploaded_file = st.sidebar.file_uploader("学習用のCSVファイルをアップロードしてください", type=['csv'])

# --- セッション状態の初期化 ---
if 'best_params' not in st.session_state:
    st.session_state.best_params = {}
if 'profile_report' not in st.session_state:
    st.session_state.profile_report = None

if uploaded_file is not None:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.profile_report = None
        st.session_state.best_params = {}
        st.session_state.last_uploaded_file = uploaded_file.name
        
    df = pd.read_csv(uploaded_file, index_col=0)
    
    with st.container(border=True):
        st.header("1. アップロードされたデータのプレビュー")
        st.dataframe(df.head())
        '''
        st.header("2. データ全体の自動分析")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("データ分析レポートを生成/再表示", help="データの統計情報、欠損値、相関などを詳細に分析します。"):
                with st.spinner("レポートを生成中です..."):
                    profile = ProfileReport(df, title="データ分析レポート", explorative=True)
                    st.session_state.profile_report = profile
                    st.rerun()
        with col2:
            if st.session_state.profile_report is not None:
                if st.button("レポートを隠す"):
                    st.session_state.profile_report = None
                    st.rerun()

    if st.session_state.profile_report is not None:
        with st.expander("📊 データ分析レポート (クリックで開閉)", expanded=True):
            st.markdown("""
            #### 📖 レポートの読み方ガイド
            このレポートは、あなたのデータの「健康診断書」です。機械学習を始める前に、データがどのような状態かを確認しましょう。
            """)
            st.markdown("##### 1. 概要 (Overview)")
            st.write("""
            データ全体のサマリーです。まずはここで全体像を掴みます。
            - **チェックポイント**:
                - `Number of variables (列数)` / `Number of observations (行数)`: データの規模が想定通りか確認します。
                - `Missing cells (欠損値)`: データ全体で、どれくらい値が欠けているかの割合です。この値が高い場合、注意が必要です。
                - `Duplicate rows (重複行)`: 全く同じ行がいくつあるかを示します。意図しない重複は、データクリーニングの対象になることがあります。
            """)
            st.markdown("##### 2. 各変数の詳細 (Variables)")
            st.write("""
            各列（変数）ごとの、より詳細な分析結果です。ここが分析のメインになります。
            - **【数値変数で見るべき点】**:
                - `mean (平均)`, `min (最小)`, `max (最大)`: 値が現実的な範囲に収まっているか確認します（例：年齢が200歳になっていないか）。極端な「外れ値」を発見する手がかりになります。
                - `Histogram (分布図)`: データの分布の形を確認します。山が一つの綺麗な形か、左右に偏っているか、二つ以上の山があるかなど、データの個性が分かります。
            - **【質的変数で見るべき点】**:
                - `Categories (カテゴリの種類と数)`: どのようなカテゴリが存在し、何種類あるかを確認します。
                - `Bar chart (棒グラフ)`: 各カテゴリの出現回数です。特定のカテゴリにデータが極端に偏っている「不均衡データ」でないかを確認できます。
            """)
            st.markdown("##### 3. 相関 (Correlations)")
            st.write("""
            変数同士の関係の強さを、ヒートマップ（色の濃淡で表現した表）で示します。
            - **チェックポイント**:
                - **目的変数との相関**: あなたが予測したい「目的変数」の行（または列）を見て、特に色が濃い（赤や青）変数を見つけましょう。それらは、予測の重要な手がかりになる可能性が高いです。
                - **説明変数同士の相関**: 説明変数同士で、非常に色が濃い（相関係数が0.9以上など）ペアはないか確認します。強すぎる相関を持つペアは、似たような情報を持っており、片方を学習から除外する候補になることがあります（多重共線性）。
            """)
            st.markdown("##### 4. 欠損値 (Missing values)")
            st.write("""
            どの列の、どのあたりにデータが欠けているかを視覚的に示します。
            - **チェックポイント**:
                - 欠損が特定の列に集中していないか、それとも全体的にまばらに発生しているかを確認します。
                - このレポートでは欠損値は修正されませんが、後の機械学習で「この列の欠損はどう扱おうか」と考えるための重要な情報になります。
            """)
            st.write("---")
            st_profile_report(st.session_state.profile_report)
            '''
            
    st.write("---")
    st.info("データを確認したら、サイドバーで学習の設定を行なってください。", icon="👇")
    
    st.sidebar.subheader("2. 目的変数と問題タイプの選択")
    all_columns = df.columns.tolist()
    target_column = st.sidebar.selectbox('目的変数を選んでください', all_columns)
    is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    default_index = 0 if is_numeric else 1
    user_choice = st.sidebar.radio("このタスクの種類を選択してください", ('回帰', '分類'), index=default_index, horizontal=True)
    problem_type = 'regression' if user_choice == '回帰' else 'classification'
    
    st.sidebar.subheader("3. 説明変数の選択")
    feature_columns = st.sidebar.multiselect(
        '予測に使う項目（説明変数）を選んでください', all_columns,
        default=[col for col in all_columns if col != target_column]
    )
    
    st.sidebar.subheader("4. 質的変数の指定 (オプション)")
    categorical_features_auto = [col for col in feature_columns if df[col].dtype == 'object']
    categorical_features_final = st.sidebar.multiselect(
        '質的変数として扱う列を選んでください', options=feature_columns, default=categorical_features_auto,
        help="数値データでもIDやランク等を表す場合はここで選択してください。"
    )

    st.sidebar.subheader("5. ハイパーパラメータ調整")
    with st.sidebar.expander("🔽 手動調整", expanded=True):
        learning_rate = st.number_input('学習率', 0.01, 1.0, st.session_state.best_params.get('learning_rate', 0.1), 0.01)
        n_estimators = st.slider('木の数', 50, 1000, st.session_state.best_params.get('n_estimators', 100), 50)
        max_depth = st.slider('木の深さ', 3, 50, st.session_state.best_params.get('max_depth', 7), 1)
        num_leaves = st.slider('葉の数', 10, 100, st.session_state.best_params.get('num_leaves', 31), 1)
        subsample = st.slider('行サンプリング率', 0.1, 1.0, st.session_state.best_params.get('subsample', 0.8), 0.1)
        colsample_bytree = st.slider('列サンプリング率', 0.1, 1.0, st.session_state.best_params.get('colsample_bytree', 0.8), 0.1)
        reg_lambda = st.number_input('L2正則化', 0.0, 10.0, st.session_state.best_params.get('reg_lambda', 1.0), 0.1)
        reg_alpha = st.number_input('L1正則化', 0.0, 10.0, st.session_state.best_params.get('reg_alpha', 0.0), 0.1)
    
    st.sidebar.subheader("6. 学習の実行")
    cv_splits = st.sidebar.slider('交差検証の分割数 (CV splits)', min_value=3, max_value=10, value=5, step=1)

    with st.sidebar.expander("🤖 パラメータ自動最適化 (Optuna)"):
        n_trials = st.number_input("試行回数 (Trials)", 10, 1000, 50, 10, help="Optunaがパラメータの組み合わせを試す回数です。")
        if st.button("最適化スタート", key="optuna_run", help="最適なハイパーパラメータの組み合わせを自動で探索します。"):
            
            def objective(trial, X_data, y_data):
                params = {
                    'random_state': 42,
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
                    'verbose': -1
                }
                scores = []
                le_obj = LabelEncoder()
                if problem_type == 'classification':
                    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    y_encoded = le_obj.fit_transform(y_data)
                    y_for_split = y_encoded
                else:
                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                    y_for_split = y_data

                for train_index, val_index in kfold.split(X_data, y_for_split):
                    X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
                    y_train_fold, y_val_fold = y_data.iloc[train_index], y_data.iloc[val_index]
                    
                    if problem_type == 'regression':
                        model = lgb.LGBMRegressor(**params); model.fit(X_train, y_train_fold)
                        score = r2_score(y_val_fold, model.predict(X_val))
                    else:
                        le_fold = LabelEncoder(); y_train_encoded_fold = le_fold.fit_transform(y_train_fold)
                        model = lgb.LGBMClassifier(**params); model.fit(X_train, y_train_encoded_fold)
                        y_val_encoded_fold = le_fold.transform(y_val_fold); score = accuracy_score(y_val_encoded_fold, model.predict(X_val))
                    scores.append(score)
                return np.mean(scores)

            with st.spinner(f"{n_trials}回の試行で最適なパラメータを探索中..."):
                features_to_use_for_optuna = feature_columns.copy()
                if target_column in features_to_use_for_optuna: features_to_use_for_optuna.remove(target_column)
                
                X_for_optuna = df[features_to_use_for_optuna].copy()
                y_for_optuna = df[target_column]
                for col in categorical_features_final:
                    if col in X_for_optuna.columns: X_for_optuna[col] = X_for_optuna[col].astype('category')
                
                study = optuna.create_study(direction='maximize')
                progress_bar = st.progress(0, text="最適化の進捗")
                status_text = st.empty()
                def callback(study, trial):
                    progress = (trial.number + 1) / n_trials
                    progress_bar.progress(progress)
                    status_text.text(f"Trial {trial.number+1}/{n_trials} | 最新スコア: {trial.value:.4f}")
                
                study.optimize(lambda trial: objective(trial, X_for_optuna, y_for_optuna), n_trials=n_trials, callbacks=[callback])
            
            st.session_state.best_params = study.best_params
            st.success(f"最適化完了！ベストスコア: {study.best_value:.4f}")
            st.info("見つかった最適パラメータが「手動調整」の各項目にセットされました。")
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button('✨ 学習スタート (手動設定)', key="manual_run"):
        with st.spinner('交差検証と最終モデルの学習を実行中です...'):
            features_to_use_manual = feature_columns.copy()
            if target_column in features_to_use_manual:
                features_to_use_manual.remove(target_column)
                st.warning(f"説明変数から目的変数 '{target_column}' を除外しました。")
            
            X = df[features_to_use_manual].copy()
            y = df[target_column]
            for col in categorical_features_final:
                if col in X.columns: X[col] = X[col].astype('category')

            params = {
                'random_state': 42, 'learning_rate': learning_rate, 'n_estimators': n_estimators,
                'max_depth': max_depth, 'num_leaves': num_leaves, 'subsample': subsample,
                'colsample_bytree': colsample_bytree, 'reg_lambda': reg_lambda, 'reg_alpha': reg_alpha
            }

            scores = []
            le = LabelEncoder()
            if problem_type == 'classification':
                kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
                y_encoded = le.fit_transform(y)
                y_for_split = y_encoded
            else:
                kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
                y_for_split = y

            for fold, (train_index, val_index) in enumerate(kfold.split(X, y_for_split)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                if problem_type == 'regression':
                    model_cv = lgb.LGBMRegressor(**params, verbose=-1)
                    model_cv.fit(X_train, y_train)
                    score = r2_score(y_val, model_cv.predict(X_val))
                else:
                    le_fold = LabelEncoder()
                    y_train_encoded_fold = le_fold.fit_transform(y_train)
                    y_val_encoded_fold = le_fold.transform(y_val)
                    model_cv = lgb.LGBMClassifier(**params, verbose=-1)
                    model_cv.fit(X_train, y_train_encoded_fold)
                    score = accuracy_score(y_val_encoded_fold, model_cv.predict(X_val))
                scores.append(score)
            
            if problem_type == 'regression':
                final_model = lgb.LGBMRegressor(**params)
                final_model.fit(X, y)
            else:
                final_model = lgb.LGBMClassifier(**params)
                final_model.fit(X, y_encoded)
                st.session_state['label_encoder'] = le

            fig, ax = plt.subplots(figsize=(12, 10))
            lgb.plot_importance(final_model, ax=ax, importance_type='gain', max_num_features=20)
            ax.tick_params(axis='y', labelsize=21)
            plt.tight_layout()

            st.session_state['model'] = final_model
            st.session_state['target_column'] = target_column
            st.session_state['problem_type'] = problem_type
            st.session_state['cv_scores'] = scores
            st.session_state['feature_columns'] = features_to_use_manual
            st.session_state['categorical_features'] = categorical_features_final
            st.session_state['feature_importance_fig'] = fig
            
            # 古い予測結果を削除
            if 'df_predict_with_results' in st.session_state:
                del st.session_state['df_predict_with_results']
            if 'prediction_score_metric' in st.session_state:
                del st.session_state['prediction_score_metric']
            
            st.balloons()
        
if 'cv_scores' in st.session_state:
    st.write("---")
    st.header("📝 モデルの成績表")
    mean_score = np.mean(st.session_state['cv_scores'])
    std_score = np.std(st.session_state['cv_scores'])
    score_metric_name = "平均 決定係数 (R2)" if st.session_state['problem_type'] == 'regression' else "平均 正解率 (Accuracy)"
    st.metric(score_metric_name, f"{mean_score:.4f}", f"± {std_score:.4f} (標準偏差)")
    with st.expander("各分割での詳細スコアを見る"):
        scores = st.session_state['cv_scores']
        chart_data = pd.DataFrame({"Score": scores}, index=[f"Fold {i+1}" for i in range(len(scores))])
        st.bar_chart(chart_data, y="Score", height=300)
        score_text = ", ".join([f"Fold {i+1}: {score:.4f}" for i, score in enumerate(scores)])
        st.caption(score_text)

if 'feature_importance_fig' in st.session_state:
    st.header("🔑 予測のカギとなった情報")
    st.pyplot(st.session_state['feature_importance_fig'])

# --- 予測パート ---
if 'model' in st.session_state:
    st.write("---")
    st.header("🔮 未来を予測してみよう！")
    
    predict_file = st.file_uploader("予測したいCSVファイルをアップロードしてください", type=['csv'], key='predict')

    if predict_file is not None:
        df_predict_original = pd.read_csv(predict_file, index_col=0)
        
        if st.button('🚀 予測を実行'):
            with st.spinner('予測を実行し、AIの思考回路を分析中です...'):
                model = st.session_state['model']
                feature_columns = st.session_state['feature_columns']
                categorical_features = st.session_state['categorical_features']
                problem_type = st.session_state['problem_type']
                target_column = st.session_state.get('target_column')

                X_pred = df_predict_original[feature_columns].copy()
                for col in categorical_features:
                    if col in X_pred.columns:
                        X_pred[col] = X_pred[col].astype('category')
                
                prediction_raw = model.predict(X_pred)
                df_predict_result = df_predict_original.copy()

                if problem_type == 'classification':
                    le = st.session_state['label_encoder']
                    prediction = le.inverse_transform(prediction_raw)
                else:
                    prediction = prediction_raw
                
                df_predict_result['予測結果'] = prediction

                if target_column and target_column in df_predict_result.columns:
                    y_true = df_predict_result[target_column]
                    if problem_type == 'classification':
                        score = accuracy_score(y_true, prediction)
                        st.session_state['prediction_score_metric'] = ("正解率 (Accuracy)", f"{score:.4f}")
                    else:
                        score = r2_score(y_true, prediction)
                        st.session_state['prediction_score_metric'] = ("決定係数 (R2 Score)", f"{score:.4f}")
                    
                    cols = df_predict_result.columns.tolist()
                    cols.insert(cols.index('予測結果') + 1, cols.pop(cols.index(target_column)))
                    df_predict_result = df_predict_result[cols]
                else:
                    if 'prediction_score_metric' in st.session_state:
                        del st.session_state['prediction_score_metric']

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_pred)
                
                st.session_state['df_predict_with_results'] = df_predict_result
                st.session_state['shap_explainer'] = explainer
                st.session_state['shap_values'] = shap_values
                st.session_state['X_pred_for_shap'] = X_pred
                st.rerun()

    if 'prediction_score_metric' in st.session_state:
        metric_label, metric_value = st.session_state['prediction_score_metric']
        st.metric(label=f"🎯 予測の答え合わせ結果", value=metric_value, help=f"予測結果と「{metric_label}」の正解データを比較したスコアです。")

    if 'df_predict_with_results' in st.session_state:
        st.subheader("予測結果")
        st.dataframe(st.session_state['df_predict_with_results'])
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')

        csv = convert_df(st.session_state['df_predict_with_results'])
        st.download_button(
            label="予測結果をCSVでダウンロード",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )
        
        if 'shap_values' in st.session_state:
            st.write("---")
            st.subheader("個別の予測の「理由」を分析 (SHAP)")
            
            shap_target_index = st.selectbox(
                "分析したい行の番号（インデックス）を選択してください",
                options=st.session_state['X_pred_for_shap'].index
            )
            
            if shap_target_index is not None:
                explainer = st.session_state['shap_explainer']
                shap_values = st.session_state['shap_values']
                X_pred_for_shap = st.session_state['X_pred_for_shap']
                shap_target_iloc = X_pred_for_shap.index.get_loc(shap_target_index)

                st.markdown("""
                <div style="padding: 1rem; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 10px;">
                <h5>📈 グラフの読み方</h5>
                <p>このグラフは、AIの予測結果（<span style="color: #ff0d57;"><b>赤色</b></span>の太字）に対して、どの情報がどのように影響したかを示します。</p>
                <ul>
                    <li><b style="color: #ff0d57;">プラスに働いた情報（赤色）</b>：予測値を押し上げる方向に影響しました。</li>
                    <li><b style="color: #008bfb;">マイナスに働いた情報（青色）</b>：予測値を押し下げる方向に影響しました。</li>
                </ul>
                <p><code>base value</code>は、何も情報がないときの平均的な予測値です。</p>
                </div>
                """, unsafe_allow_html=True)
                st.write("")

                if st.session_state['problem_type'] == 'classification':
                    if isinstance(shap_values, list):
                        st.info("分類問題では、陽性クラス（例: Survived=1）に対する各特徴量の影響度を表示します。")
                        st_shap(shap.force_plot(
                            explainer.expected_value[1],
                            shap_values[1][shap_target_iloc],
                            X_pred_for_shap.iloc[shap_target_iloc]
                        ))
                    else:
                        st.info("分類モデルの出力に対する各特徴量の影響度を表示します。")
                        st_shap(shap.force_plot(
                            explainer.expected_value,
                            shap_values[shap_target_iloc],
                            X_pred_for_shap.iloc[shap_target_iloc]
                        ))
                else:
                    st_shap(shap.force_plot(
                        explainer.expected_value,
                        shap_values[shap_target_iloc],
                        X_pred_for_shap.iloc[shap_target_iloc]
                    ))

else:
    with st.expander("🤔 アプリの使いかたガイド", expanded=True):
        st.markdown("""
        ### 1. データの準備
        - 予測したい「答え」の列を含む、CSV形式の学習用データをご用意ください。
        - 必要であれば、「サンプルデータ置き場」ページのデータをダウンロードしてお使いいただけます。
        ### 2. データと設定
        - サイドバーの「参照」ボタンから、学習用CSVデータをアップロードします。
        - **目的変数**: 予測したい「答え」の列を指定します。
        - **タスクの種類**: 目的変数が数値なら「回帰」、カテゴリなら「分類」を選びます。
        - **ハイパーパラメータ**: AIモデルの学習方法を細かく調整できます。または「自動最適化」でAIに探させることもできます。
        - **説明変数**: 予測の手がかりとなる列を選びます。
        - **質的変数**: 説明変数の中で、数値でもカテゴリとして扱いたい列があれば指定します。
        ### 3. 学習と分析
        - 「学習スタート」ボタンを押すと、モデルの評価と学習が始まります。
        - **モデルの成績表**: 作成されたモデルの賢さを示すスコアです。
        - **予測のカギとなった情報**: どの変数が予測に重要だったかを示します。
        ### 4. 未来の予測
        - 「未来を予測してみよう！」セクションで、答えの列がない新しいCSVをアップロードし、予測を実行します。
        """)