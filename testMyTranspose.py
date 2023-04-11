import unittest

import numpy as np
import pandas as pd

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
        
    def testTranspose3(self) :
        # Generate Array
        myVar1 = np.array([1,2]).reshape(1,2)
        
        # Run Inference
        myVar1Pred = myTranspose(myVar1)
        myVar1Target = myVar1.transpose()
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar1Pred, myVar1Target) is None)
        
    def testTranspose4(self) :
        # Generate Array
        myVar1 = np.array([1,2]).reshape(2,1)
        
        # Run Inference
        myVar1Pred = myTranspose(myVar1)
        myVar1Target = myVar1.transpose()
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar1Pred, myVar1Target) is None)
        
    def testTranspose5(self) :
        # Generate Array
        myVar2 = np.array([1,2,np.nan,4])
        
        # Run Inference
        myVar2Pred = myTranspose(myVar2)
        myVar2Target = myVar2.transpose()
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar2Pred, myVar2Target) is None)
    
    def testTranspose6(self) :
        # Generate Array
        myVar2 = np.array([np.nan])
        
        # Run Inference
        myVar2Pred = myTranspose(myVar2)
        myVar2Target = myVar2.transpose()
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar2Pred, myVar2Target) is None)
    
    def testTranspose7(self) :
        # Generate Array
        myVar2 = np.nan
        
        # Run Inference
        myVar2Pred = myTranspose(myVar2)
        myVar2Target = np.nan
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myVar2Pred, myVar2Target) is None)
    
    def testTranspose8(self) :
        # Generate DataFrame
        d, e, f = np.array([1,2,3,4]), np.array(["red","white","red",np.nan]), np.array([True,True,True,False])
        myData3 = pd.DataFrame([d,e,f]).transpose()
        
        # Run Inference
        myData3Pred = myTranspose(myData3)
        myData3Target = np.array(myData3.transpose())
        
        # Unit Test
        self.assertTrue(np.testing.assert_equal(myData3Pred, myData3Target) is None)


if __name__ == "__main__" :
    unittest.main()
