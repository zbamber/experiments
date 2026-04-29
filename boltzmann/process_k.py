import numpy as np
import pandas as pd

m = np.array([4.1884e1,4.1673e1,3.0970e1,2.9868e1,3.0793e1])
m_uncertainty = np.array([3.3352e-1,3.6378e-1,8.0203e-2,1.0727e-1,9.7058e-2])
temps = np.array([0,0,100,100,100])


inverse_m = 1 / m
inverse_m_error = m_uncertainty / m**2
output = pd.DataFrame({
        'temp': temps,
        '1/m': inverse_m,
        '1/m_error': inverse_m_error
    })
output.to_csv('k_processed.csv', index=False, header=False)