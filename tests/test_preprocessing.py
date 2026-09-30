from src.data.data_preprocessing import build_preprocessor
def test_preprocessor(): assert 'num' in build_preprocessor().named_transformers
