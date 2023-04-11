import numpy as np
import pandas as pd


def myTranspose(inputArray) :
    if isinstance(inputArray, float) :
        return inputArray
    elif isinstance(inputArray, pd.DataFrame) :
        # Get Input DataFrame Shape
        numRow, numCol = inputArray.shape
        
        # Add Numpy Array
        dataList = []
        for i in range(numCol) :
            dataList.append(np.array(inputArray.loc[:, i]))

        return np.array(dataList)
    else :
        if len(inputArray.shape) == 1 :
            # Get Input Array Shape
            numRow = inputArray.shape[0]
            
            # Create Dummy Array
            inputArrayTP = np.ones(numRow)
            
            # Transpose Array
            for i in range(numRow) :
                inputArrayTP[i] = inputArray[i]
        else :
            # Get Input Array Shape
            numRow, numCol = inputArray.shape

            # Create Dummy Array
            inputArrayTP = np.ones((numCol, numRow))
            
            # Transpose Array
            for i in range(numRow) :
                for j in range(numCol) :
                    inputArrayTP[j,i] = inputArray[i,j]

        return inputArrayTP


if __name__ == "__main__" :
    np.random.seed(42)
    x = np.random.randn(4,2)
    y = myTranspose(x)
    print(x)
    print(y)
