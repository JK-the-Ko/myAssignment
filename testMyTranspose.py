import unittest

import numpy as np

from myTranspose import myTranspose


class myTestCase(unittest.TestCase) :
    def testTranspose1(self) :
        # Generate Array
        myVar1 = np.arange(1,11).reshape(2,5).transpose()
        
        # Run Inference
        myVar1Pred = myTranspose(myVar1)
        myVar1Target = myVar1.transpose()
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar1Pred, myVar1Target) is None)
        
    def testTranspose2(self) :
        # Generate Array
        myVar1 = np.ones(shape=(0,0))*np.nan
        
        # Run Inference
        myVar1Pred = myTranspose(myVar1)
        myVar1Target = np.empty(shape=(0,0))
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar1Pred, myVar1Target) is None)


if __name__ == "__main__" :
    unittest.main()
