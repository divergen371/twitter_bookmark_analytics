from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["tweeted_at", "screen_name", "full_text"]

TECH_KEYWORDS = [
    "python",
    "javascript",
    "programming",
    "code",
    "developer",
    "tech",
    "cybersecurity",
    "security",
    "it",
    "software",
    "web",
    "api",
    "data",
    "github",
    "docker",
    "cloud",
    "ai",
    "ml",
    "database",
    "dev",
    "coding",
    "プログラミング",
    "エンジニア",
    "コード",
    "セキュリティ",
    "機械学習",
    "クラウド",
    "クラウドコンピューティング",
    "キャリア",
    "セキュリティ対策"
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """ブックマークデータを読み込んで前処理を行う.

    Args:
        file_path: ブックマークデータを含むCSVファイルのパス

    Returns:
        pd.DataFrame: 前処理済みのブックマークデータ

    Raises:
        FileNotFoundError: 指定されたファイルが存在しない場合
        pd.errors.EmptyDataError: ファイルが空の場合
        pd.errors.ParserError: ファイル形式が不正な場合
        ValueError: 必須カラムが不足している場合

    """
    logger.info("データファイルを読み込みます: %s", file_path)

    # まずヘッダー行のみを読み込んで必須カラムの存在を確認
    try:
        header_df = pd.read_csv(file_path, nrows=0)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in header_df.columns]
        if missing_columns:
            # 必須カラムが不足している場合はParserErrorを発生させる
            raise pd.errors.ParserError(f"必須カラムが不足しています: {', '.join(missing_columns)}")
    except pd.errors.EmptyDataError:
        logger.exception("ファイルが空です: %s", file_path)
        raise
    except pd.errors.ParserError:
        logger.exception("ファイルの形式が不正です")
        raise
    except FileNotFoundError:
        logger.exception("ファイルが見つかりません: %s", file_path)
        raise
    except Exception as e:
        logger.exception("ファイルの解析中にエラーが発生しました")
        raise pd.errors.ParserError(str(e)) from e

    try:
        # 全データを読み込む
        bookmark_df = pd.read_csv(
            file_path,
            on_bad_lines='error',
            delimiter=','
        )
    except FileNotFoundError:
        logger.exception("ファイルが見つかりません: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logger.exception("ファイルが空です: %s", file_path)
        raise
    except pd.errors.ParserError:
        logger.exception("ファイルの形式が不正です")
        raise
    except Exception:
        logger.exception("予期せぬエラーが発生しました")
        raise

    try:
        bookmark_df["tweeted_at"] = pd.to_datetime(bookmark_df["tweeted_at"])
    except Exception as e:
        error_msg = f"日時データの変換に失敗しました: {e}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e

    logger.info("データの読み込みが完了しました。レコード数: %d", len(bookmark_df))
    return bookmark_df


def categorize_tech(text: str) -> str:
    """テキストをテクノロジー関連かどうかで分類する.

    Args:
        text: 分類対象のテキスト

    Returns:
        str: カテゴリ("テクノロジー" または "その他")

    """
    if not isinstance(text, str):
        logger.warning("テキストが文字列ではありません: %s", type(text))
        return "その他"
    if pd.isna(text):
        logger.debug("テキストが空値です")
        return "その他"

    text_lower = text.lower()
    return "テクノロジー" if any(keyword in text_lower for keyword in TECH_KEYWORDS) else "その他"
