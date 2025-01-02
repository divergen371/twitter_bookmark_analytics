import pandas as pd
import pytest

from twitter_bookmark_analytics.utils.data_logger import categorize_tech, load_data


# load_dataのテスト
def test_load_data_valid(tmp_path):
    """正常なCSVファイルの読み込みテスト"""
    # テスト用のCSVファイルを作成
    test_csv = tmp_path / "test.csv"
    df = pd.DataFrame({
        "tweeted_at": ["2023-01-01 12:00:00", "2023-01-02 12:00:00"],
        "screen_name": ["user1", "user2"],
        "full_text": ["テストツイート1", "テストツイート2"]
    })
    df.to_csv(test_csv, index=False)
    
    # テスト実行
    result = load_data(test_csv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(col in result.columns for col in ["tweeted_at", "screen_name", "full_text"])
    assert pd.api.types.is_datetime64_any_dtype(result["tweeted_at"])

def test_load_data_file_not_found():
    """存在しないファイルを指定した場合のテスト"""
    with pytest.raises(FileNotFoundError):
        load_data("not_exists.csv")

def test_load_data_empty_file(tmp_path):
    """空のファイルを指定した場合のテスト"""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    
    with pytest.raises(pd.errors.EmptyDataError):
        load_data(empty_csv)

def test_load_data_invalid_format(tmp_path):
    """不正な形式のファイルを指定した場合のテスト"""
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("invalid,csv,format\n1,2")
    
    with pytest.raises(pd.errors.ParserError):
        load_data(invalid_csv)

def test_load_data_missing_columns(tmp_path):
    """必須カラムが不足している場合のテスト"""
    test_csv = tmp_path / "missing_columns.csv"
    df = pd.DataFrame({
        "tweeted_at": ["2023-01-01 12:00:00"],
        "screen_name": ["user1"]
        # full_textカラムが欠落
    })
    df.to_csv(test_csv, index=False)
    
    with pytest.raises(ValueError, match="必須カラムが不足しています"):
        load_data(test_csv)

def test_load_data_invalid_datetime(tmp_path):
    """不正な日時形式の場合のテスト"""
    test_csv = tmp_path / "invalid_datetime.csv"
    df = pd.DataFrame({
        "tweeted_at": ["invalid_datetime"],
        "screen_name": ["user1"],
        "full_text": ["テストツイート"]
    })
    df.to_csv(test_csv, index=False)
    
    with pytest.raises(ValueError, match="日時データの変換に失敗しました"):
        load_data(test_csv)


# categorize_techのテスト
def test_categorize_tech_technology():
    """テクノロジー関連のテキストを正しく分類できるかテスト"""
    tech_texts = [
        "Pythonプログラミングについて",
        "JavaScriptの新機能",
        "GitHubの使い方",
        "機械学習とAI",
        "クラウドコンピューティング",
        "エンジニアのキャリア",
        "セキュリティ対策"
    ]
    for text in tech_texts:
        assert categorize_tech(text) == "テクノロジー"

def test_categorize_tech_non_technology():
    """非テクノロジー関連のテキストを正しく分類できるかテスト"""
    non_tech_texts = [
        "今日の天気について",
        "美味しいレストラン",
        "旅行の計画",
        "スポーツニュース"
    ]
    for text in non_tech_texts:
        assert categorize_tech(text) == "その他"

def test_categorize_tech_empty_input():
    """空の入力に対するテスト"""
    assert categorize_tech("") == "その他"
    assert categorize_tech(None) == "その他"

def test_categorize_tech_non_string():
    """文字列以外の入力に対するテスト"""
    assert categorize_tech(123) == "その他"
    assert categorize_tech(True) == "その他"
    assert categorize_tech([]) == "その他"
