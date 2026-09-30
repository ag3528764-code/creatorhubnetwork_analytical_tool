from src.data.data_generator import generate_dataset
def test_shape(): assert generate_dataset(100,42).shape==(100,14)
def test_target(): assert set(generate_dataset(100,42).purchased.unique())<= {0,1}
