import unittest
from algosciquant import *

"""
unittest tutorial
    https://www.blog.pythonlibrary.org/2016/07/07/python-3-testing-an-intro-to-unittest/
Run this module on the command line
   > python3 algosciquant_t.py
"""

class TestAdd(unittest.TestCase):
    """
    Test ndayTruth function
    """

    def test_ndayTruth(self):
        n=3

        df_test = pd.DataFrame(data={'close_price': [1.0, 1.1, 1.0, 1.1, 0.9, 1.1, 1.0]},
                               index=[dt.datetime(2017, 1, 1), dt.datetime(2017, 1, 2),
                                      dt.datetime(2017, 1, 3), dt.datetime(2017, 1, 4),
                                      dt.datetime(2017, 1, 5), dt.datetime(2017, 1, 8),
                                      dt.datetime(2017, 1, 9)])

        x = pd.DataFrame(data={'x_n': [1.0, -1.0, 1.0, -1.0, np.nan, np.nan, np.nan]},
                         index=[dt.datetime(2017, 1, 1), dt.datetime(2017, 1, 2),
                                dt.datetime(2017, 1, 3), dt.datetime(2017, 1, 4),
                                dt.datetime(2017, 1, 5), dt.datetime(2017, 1, 8),
                                dt.datetime(2017, 1, 9)])

        result = ndayTruth(df_test, nday=n, tvariable='close_price')

        for k in range(0,len(df_test-n)-n):
            self.assertEqual(result.iloc[k]['t_n'], x.iloc[k]['x_n'])

if __name__ == '__main__':
    unittest.main()