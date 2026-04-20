import pandas as pd
import numpy as np
import glob

VOLTAGE_UNCERTAINTY = 0.028
fileList = [f for f in glob.glob('Data/*.csv') if '_processed' not in f]

for fileName in fileList:
    data = pd.read_csv(fileName, usecols=[3,4], names=['time', 'voltage'], header=None)
    times = data['time'].to_numpy()
    voltages = data['voltage'].to_numpy()

    peaks = np.where(
        (voltages[1:-1] >= voltages[:-2]) &
        (voltages[1:-1] > voltages[2:]) &
        (voltages[1:-1] > 0)
    )[0] + 1

    peakTimes = times[peaks]
    Amplitudes = voltages[peaks]

    lnAmplitudes = np.log(Amplitudes)
    deltaLnAmplitudes = VOLTAGE_UNCERTAINTY / Amplitudes

    output = pd.DataFrame({
        'Time': peakTimes,
        'ln(Amplitude)': lnAmplitudes,
        'Uncertainty': deltaLnAmplitudes
    })
    fileName = fileName.replace('.CSV', '_processed.csv')
    output.to_csv(fileName, index=False, header=False)