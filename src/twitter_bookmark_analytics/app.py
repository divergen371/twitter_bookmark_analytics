import logging  # noqa: I001
import sys
from pathlib import Path

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st

from twitter_bookmark_analytics.utils.data_logger import categorize_tech, load_data
from twitter_bookmark_analytics.utils.visualizations import (
    create_category_plot,
    create_time_series_plot,
    get_top_words,
)

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def main():
    logger.info("アプリケーションを開始します")
    st.title("Twitter ブックマーク分析ダッシュボード")

    try:
        data_path = Path(__file__).parent / "data" / "bookmarks.csv"
        logger.info("データファイルを読み込みます: %s", data_path)
        bookmark_df = load_data(data_path)
    except FileNotFoundError:
        error_msg = f"ブックマークデータファイルが見つかりません: {data_path!s}"
        logger.exception(error_msg)
        st.error(error_msg)
        return
    except pd.errors.ParserError as e:
        error_msg = (
            f"ブックマークデータファイルの読み込みに失敗しました: {data_path!s} ファイル形式が不正です。詳細: {e!s}"
        )
        logger.exception(error_msg)
        st.error(error_msg)
        return
    except pd.errors.EmptyDataError:
        error_msg = f"ブックマークデータファイルが空です: {data_path!s}"
        logger.exception(error_msg)
        st.error(error_msg)
        return
    except Exception as e:
        error_msg = f"予期せぬエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        st.error(error_msg)
        return

    logger.info("データの読み込みに成功しました。レコード数: %d", len(bookmark_df))

    try:
        bookmark_df["category"] = bookmark_df["full_text"].apply(categorize_tech)
        logger.info("カテゴリ分類が完了しました")
    except Exception as e:
        error_msg = f"カテゴリ分類中にエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        st.error(error_msg)
        return

    st.sidebar.header("フィルター")

    try:
        date_range = st.sidebar.date_input(
            "日付範囲を選択",
            value=(bookmark_df["tweeted_at"].min().date(), bookmark_df["tweeted_at"].max().date()),
            min_value=bookmark_df["tweeted_at"].min().date(),
            max_value=bookmark_df["tweeted_at"].max().date(),
        )
        logger.info("日付範囲が選択されました: %s", date_range)
    except Exception:
        error_msg = "日付範囲の設定中にエラーが発生しました"
        logger.exception(error_msg)
        st.error(error_msg)
        return

    selected_categories = st.sidebar.multiselect(
        "カテゴリを選択",
        options=bookmark_df["category"].unique(),
        default=bookmark_df["category"].unique(),
    )

    mask = (
        (bookmark_df["tweeted_at"].dt.date >= date_range[0])
        & (bookmark_df["tweeted_at"].dt.date <= date_range[1])
        & (bookmark_df["category"].isin(selected_categories))
    )
    filtered_df = bookmark_df[mask]

    st.header("基本統計")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("総ブックマーク数", len(filtered_df))
    with col2:
        st.metric("テクノロジー関連の割合", f"{(filtered_df["category"] == "テクノロジー").mean():.1%}")
    with col3:
        st.metric("ユニークユーザー数", filtered_df["screen_name"].nunique())

    st.header("時系列分析")
    try:
        time_series_plot = create_time_series_plot(filtered_df)
        st.plotly_chart(time_series_plot)
        logger.info("時系列分析プロットを作成しました")
    except Exception:
        error_msg = "時系列分析プロットの作成中にエラーが発生しました"
        logger.exception(error_msg)
        st.error(error_msg)

    st.header("カテゴリ分布")
    try:
        category_plot = create_category_plot(filtered_df)
        st.plotly_chart(category_plot)
        logger.info("カテゴリ分布プロットを作成しました")
    except Exception:
        error_msg = "カテゴリ分布プロットの作成中にエラーが発生しました"
        logger.exception(error_msg)
        st.error(error_msg)

    st.header("頻出ワード分析")
    tech_only = st.checkbox("テクノロジー関連用語のみを表示", value=False)
    try:
        top_words = get_top_words(filtered_df["full_text"], tech_only=tech_only)
        title = "頻出ワード（テクノロジー関連用語のみ）" if tech_only else "頻出ワード"
        fig = px.bar(x=list(top_words.keys()), y=list(top_words.values()), title=title)
        fig.update_layout(xaxis_title="単語", yaxis_title="出現回数", xaxis={"tickangle": 45})
        st.plotly_chart(fig)
        logger.info("頻出ワード分析プロットを作成しました")
    except Exception:
        error_msg = "頻出ワード分析の作成中にエラーが発生しました"
        logger.exception(error_msg)
        st.error(error_msg)

    st.header("生データ")
    if st.checkbox("生データを表示"):
        st.dataframe(filtered_df[["tweeted_at", "screen_name", "full_text", "category"]])


if __name__ == "__main__":
    main()
