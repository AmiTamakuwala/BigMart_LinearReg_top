import pandas as pd

def concat(X,Y):
    """
    X:test datset.
    Y: train datset.
    return: None
    """
    df = pd.concat([X,Y], ignore_index=True)
    return df

print(df.shape)