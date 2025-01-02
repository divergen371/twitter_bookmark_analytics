from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pytest

from twitter_bookmark_analytics.utils.visualizations import (
    create_category_plot,
    create_time_series_plot,
    get_top_words,
    validate_dataframe,
)


# validate_dataframeのテスト
def test_validate_dataframe_valid():
    """正常なDataFrameの検証テスト"""
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    validate_dataframe(df, ['col1', 'col2'])  # エラーが発生しないことを確認

def test_validate_dataframe_invalid_type():
    """不正な型の入力テスト"""
    with pytest.raises(TypeError):
        validate_dataframe({'col1': [1, 2]}, ['col1'])  # dictを渡してTypeErrorを確認

def test_validate_dataframe_missing_columns():
    """必要なカラムが存在しない場合のテスト"""
    df = pd.DataFrame({'col1': [1, 2]})
    with pytest.raises(ValueError):
        validate_dataframe(df, ['col1', 'missing_col'])

# create_time_series_plotのテスト
def test_create_time_series_plot_valid():
    """正常な時系列プロット作成テスト"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    df = pd.DataFrame({
        'tweeted_at': dates,
        'text': ['text'] * len(dates)
    })
    fig = create_time_series_plot(df)
    assert isinstance(fig, go.Figure)

def test_create_time_series_plot_invalid_column():
    """必要なカラムが存在しない場合のテスト"""
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    with pytest.raises(ValueError):
        create_time_series_plot(df)

# create_category_plotのテスト
def test_create_category_plot_valid():
    """正常なカテゴリプロット作成テスト"""
    df = pd.DataFrame({
        'category': ['Tech', 'Non-Tech', 'Tech', 'Tech', 'Non-Tech']
    })
    fig = create_category_plot(df)
    assert isinstance(fig, go.Figure)

def test_create_category_plot_invalid_column():
    """必要なカラムが存在しない場合のテスト"""
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    with pytest.raises(ValueError):
        create_category_plot(df)

# get_top_wordsのテスト
def test_get_top_words_valid():
    """正常な頻出単語分析テスト"""
    texts = [
        "プログラミング言語のPythonについて",
        "Pythonでデータ分析をする",
        "機械学習とPythonプログラミング"
    ]
    result = get_top_words(texts, n=2)
    assert isinstance(result, dict)
    assert len(result) <= 2  # 上位2件まで

def test_get_top_words_empty_input():
    """空の入力テスト"""
    with pytest.raises(ValueError):
        get_top_words([])

def test_get_top_words_invalid_type():
    """不正な型の入力テスト"""
    with pytest.raises(TypeError):
        get_top_words(123)  # 数値を渡してTypeErrorを確認

def test_get_top_words_tech_only():
    """テクノロジー関連単語のみの抽出テスト"""
    texts = [
        "プログラミング言語のPythonについて",
        "今日は晴れです",
        "機械学習とPythonプログラミング"
    ]
    result = get_top_words(texts, n=2, tech_only=True)
    assert isinstance(result, dict)
    assert len(result) <= 2
