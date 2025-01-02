from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from twitter_bookmark_analytics.app import main


@pytest.fixture
def mock_streamlit(monkeypatch):
    """Streamlitの関数をモックするフィクスチャ"""
    # Streamlitの関数をモック化
    monkeypatch.setattr(st, "title", lambda x: None)
    monkeypatch.setattr(st, "header", lambda x: None)
    monkeypatch.setattr(st, "error", lambda x: None)
    monkeypatch.setattr(st, "sidebar", st)
    monkeypatch.setattr(st, "columns", lambda x: [type("Column", (), {"metric": lambda x, y: None})] * x)
    monkeypatch.setattr(st, "plotly_chart", lambda x: None)
    monkeypatch.setattr(st, "checkbox", lambda x, value=False: value)
    monkeypatch.setattr(st, "dataframe", lambda x: None)
    monkeypatch.setattr(st, "multiselect", lambda *args, **kwargs: kwargs.get("default", []))
    monkeypatch.setattr(st, "date_input", lambda *args, **kwargs: [kwargs.get("min_value"), kwargs.get("max_value")])


@pytest.fixture
def sample_bookmark_data(tmp_path):
    """テスト用のブックマークデータを作成するフィクスチャ"""
    data_dir = tmp_path / "src" / "twitter_bookmark_analytics" / "data"
    data_dir.mkdir(parents=True)
    
    # テスト用のCSVファイルを作成
    csv_path = data_dir / "bookmarks.csv"
    df = pd.DataFrame({
        "tweeted_at": ["2023-01-01 12:00:00", "2023-01-02 12:00:00"],
        "screen_name": ["user1", "user2"],
        "full_text": ["Pythonプログラミング", "今日の天気"]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_main_with_valid_data(mock_streamlit, sample_bookmark_data, monkeypatch):
    """正常なデータでアプリケーションが動作することを確認するテスト"""
    # __file__の値を一時的に変更
    monkeypatch.setattr(Path, "__file__", sample_bookmark_data)
    
    try:
        main()
    except Exception as e:
        pytest.fail(f"アプリケーションの実行中にエラーが発生しました: {e}")


def test_main_with_missing_data(mock_streamlit, tmp_path, monkeypatch):
    """データファイルが存在しない場合のテスト"""
    non_existent_path = tmp_path / "non_existent.csv"
    monkeypatch.setattr(Path, "__file__", str(non_existent_path))
    
    # エラーが発生せずに処理が終了することを確認
    main()


def test_main_with_empty_data(mock_streamlit, tmp_path, monkeypatch):
    """空のデータファイルの場合のテスト"""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    monkeypatch.setattr(Path, "__file__", str(empty_file))
    
    # エラーが発生せずに処理が終了することを確認
    main()


def test_main_with_invalid_data(mock_streamlit, tmp_path, monkeypatch):
    """不正なデータファイルの場合のテスト"""
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("invalid,csv,format\n1,2")
    monkeypatch.setattr(Path, "__file__", str(invalid_file))
    
    # エラーが発生せずに処理が終了することを確認
    main()


def test_main_with_missing_columns(mock_streamlit, tmp_path, monkeypatch):
    """必須カラムが不足しているデータファイルの場合のテスト"""
    missing_columns_file = tmp_path / "missing_columns.csv"
    df = pd.DataFrame({
        "tweeted_at": ["2023-01-01 12:00:00"],
        "screen_name": ["user1"]
        # full_textカラムが欠落
    })
    df.to_csv(missing_columns_file, index=False)
    monkeypatch.setattr(Path, "__file__", str(missing_columns_file))
    
    # エラーが発生せずに処理が終了することを確認
    main()
