from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path

import MeCab
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords

# NLTKのデータディレクトリを設定
nltk_data_path = Path("nltk_data").resolve
os.environ["NLTK_DATA"] = str(nltk_data_path)
nltk.data.path = [str(nltk_data_path)]  # 他のパスを上書きして、このパスのみを使用

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """データフレームの妥当性を検証する.

    Args:
        df: 検証するデータフレーム
        required_columns: 必要なカラムのリスト

    Raises:
        TypeError: 入力が不正な型の場合
        ValueError: 必要なカラムが存在しない場合

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("入力はpandas.DataFrameである必要があります")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必要なカラムが存在しません: {missing_columns}")


def create_time_series_plot(df: pd.DataFrame) -> go.Figure:
    """ツイートの時系列プロットを作成する.

    Args:
        df: ツイートデータを含むDataFrame(tweeted_atカラムが必要)

    Returns:
        go.Figure: 時系列プロット

    Raises:
        TypeError: 入力が不正な型の場合
        ValueError: 必要なカラムが存在しない場合
        pd.errors.ResampleError: 時系列データの集計に失敗した場合

    """
    logger.info("時系列プロットの作成を開始")

    try:
        validate_dataframe(df, ["tweeted_at"])
        daily_counts = df.resample("D", on="tweeted_at").size().reset_index()
        daily_counts.columns = ["date", "count"]

        fig = px.line(daily_counts, x="date", y="count", title="日別ブックマーク数の推移")
        fig.update_layout(xaxis_title="日付", yaxis_title="ブックマーク数")

    except (TypeError, ValueError) as e:
        error_msg = f"時系列プロットの作成中にエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        raise
    except pd.errors.ResampleError as e:
        error_msg = f"時系列データの集計に失敗しました: {e!s}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"予期せぬエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    else:
        logger.info("時系列プロットの作成が完了")
        return fig


def create_category_plot(df: pd.DataFrame) -> go.Figure:
    """テクノロジー関連・非関連ツイートの円グラフを作成する.

    Args:
        df: ツイートデータを含むDataFrame(categoryカラムが必要)

    Returns:
        go.Figure: カテゴリ分布の円グラフ

    Raises:
        TypeError: 入力が不正な型の場合
        ValueError: 必要なカラムが存在しない場合

    """
    logger.info("カテゴリプロットの作成を開始")

    try:
        validate_dataframe(df, ["category"])

        category_counts = df["category"].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index, title="カテゴリ別の割合")

        logger.info("カテゴリプロットの作成が完了")
    except (TypeError, ValueError) as e:
        error_msg = f"カテゴリプロットの作成中にエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        raise
    except Exception as e:
        error_msg = f"予期せぬエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    else:
        return fig


def get_top_words(texts: list[str] | pd.Series, n: int = 50, *, tech_only: bool = False) -> dict[str, int]:
    """テキストから最も頻出する単語を抽出する.

    Args:
        texts: 分析対象のテキストのリストまたはSeries
        n: 返す単語の数(デフォルト: 20)

    Returns:
        Dict[str, int]: 単語と出現回数の辞書

    Raises:
        TypeError: 入力が不正な型の場合
        ValueError: 入力が空の場合

    """
    logger.info("頻出単語分析を開始(上位%d件)", n)

    try:

        def _validate_input(texts: list[str] | pd.Series) -> None:
            """入力の型と長さを検証する内部関数."""
            if not isinstance(texts, (list, pd.Series)):
                raise TypeError("入力はリストまたはSeriesである必要があります")
            if len(texts) == 0:
                raise ValueError("入力テキストが空です")

        _validate_input(texts)

        all_text = " ".join(str(text) for text in texts if pd.notna(text))

        def _preprocess_text(text: str) -> str:
            text = re.sub(r"http\S+|www.\S+", "", text)
            return re.sub(r"[^\w\s]", "", text)

        if not all_text.strip():
            error_msg = "有効なテキストが存在しません"
            raise ValueError(error_msg)

        all_text = _preprocess_text(all_text)

        # MeCabで形態素解析を行う
        tagger = MeCab.Tagger("-Owakati")
        words = []
        for text in texts:
            if pd.notna(text):
                processed_text = _preprocess_text(str(text))
                words.extend(tagger.parse(processed_text).split())

        # NLTKの日本語ストップワードを使用
        try:
            stop_words = set(stopwords.words("japanese"))
        except LookupError:
            logger.warning("日本語のストップワードが見つかりません。デフォルトのストップワードを使用します。")
            stop_words = set()

        # Twitterに特有の不要な単語を追加
        stop_words.update(
            {
                "RT",
                "http",
                "https",
                "co",
                "jp",
                "com",
                "www",
                "amp",
                "...",
                "…",
                "！",  # noqa: RUF001
                "？",  # noqa: RUF001
                "笑",
                "w",
                "ｗ",  # noqa: RUF001
                "♪",
                "：",  # noqa: RUF001
                "；",  # noqa: RUF001
                "する",
                "いる",
                "なる",
                "ある",
                "れる",
                "の",
                "が",
                "に",
                "を",
                "は",
                "た",
                "です",
                "ます",
                "ない",
                "だ",
                "って",
                "て",
                "と",
                "も",
                "な",
            },
        )

        # テクノロジー関連用語のセット
        tech_words = {
            "AWS",
            "Azure",
            "GCP",
            "Kubernetes",
            "Docker",
            "DevOps",
            "CI/CD",
            "マイクロサービス",
            "コンテナ",
            "サーバーレス",
            "インフラ",
            "クラウド",
            "フロントエンド",
            "バックエンド",
            "フルスタック",
            "データベース",
            "SQL",
            "NoSQL",
            "API",
            "REST",
            "GraphQL",
            "WebSocket",
            "HTTP",
            "HTTPS",
            "SSL",
            "TLS",
            "セキュリティ",
            "認証",
            "認可",
            "暗号化",
            "ハッシュ",
            "アルゴリズム",
            "機械学習",
            "ディープラーニング",
            "AI",
            "人工知能",
            "ニューラル",
            "データサイエンス",
            "ビッグデータ",
            "分散処理",
            "並列処理",
            "スケーラビリティ",
            "可用性",
            "信頼性",
            "モニタリング",
            "ロギング",
            "デバッグ",
            "テスト",
            "CI",
            "CD",
            "Git",
            "GitHub",
            "GitLab",
            "BitBucket",
            "アジャイル",
            "スクラム",
            "カンバン",
            "レガシー",
            "リファクタリング",
            "クリーンコード",
            "デザインパターン",
            "アーキテクチャ",
            "モノリス",
            "サービスメッシュ",
            "Istio",
            "Envoy",
            "Prometheus",
            "Grafana",
            "ELK",
            "Elasticsearch",
            "ブロックチェーン",
            "スマートコントラクト",
            "Web3",
            "NFT",
            "メタバース",
            "AR",
            "VR",
            "IoT",
            "5G",
            "6G",
            "エッジコンピューティング",
            "フォグコンピューティング",
            "マルチクラウド",
            "ハイブリッドクラウド",
            "SaaS",
            "PaaS",
            "IaaS",
            "FaaS",
            "BaaS",
            "DaaS",
            "XaaS",
            "オートメーション",
            "RPA",
            "ローコード",
            "ノーコード",
            "PWA",
            "SPA",
            "SSR",
            "CSR",
            "JAMstack",
            "WebAssembly",
            "WASM",
            "Rust",
            "Go",
            "Python",
            "JavaScript",
            "TypeScript",
            "React",
            "Vue",
            "Angular",
            "Svelte",
            "Next.js",
            "Nuxt.js",
            "Node.js",
            "Deno",
            "Django",
            "Flask",
            "FastAPI",
            "Spring",
            "Rails",
            "Laravel",
            "PHP",
            "Java",
            "Kotlin",
            "Swift",
            "Flutter",
            "React Native",
            "Xamarin",
            "Unity",
            "Unreal Engine",
            "Clojure",
            "Elixir",
            "Scala",
            "Common Lisp",
            "Erlang",
            "python",
            "javascript",
            "programming",
            "code",
            "developer",
            "tech",
            "cybersecurity",
            "security",
            "IT",
            "software",
            "web",
            "api",
            "data",
            "github",
            "docker",
            "cloud",
            "ML",
            "database",
            "dev",
            "coding",
            "プログラミング",
            "エンジニア",
            "コード",
        }

        # 単語のカウント
        if tech_only:
            word_counts = Counter(
                word for word in words if word in tech_words and len(word) > 1 and not word.isdigit()
            )
        else:
            word_counts = Counter(
                word
                for word in words
                if word.lower() not in stop_words
                and len(word) > 1
                and not word.isdigit()
                and not all(c.isascii() for c in word)  # 英語のみの単語を除外
            )

        result = dict(word_counts.most_common(n))
        logger.info("頻出単語分析が完了")

    except (TypeError, ValueError) as e:
        error_msg = f"頻出単語分析中にエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        raise
    except Exception as e:
        error_msg = f"予期せぬエラーが発生しました: {e!s}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    else:
        return result
