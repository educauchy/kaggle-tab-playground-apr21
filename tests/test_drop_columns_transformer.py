import unittest
from Transformers.DropColumnsTransformer import DropColumnsTransformer


class TestDropColumnsTransformer(unittest.TestCase):
    def test_no_columns(self):
        obj = DropColumnsTransformer(columns=[])
        assert len(obj.columns) != 0, 'There should be at least one column!'


if __name__ == '__main__':
    unittest.main()