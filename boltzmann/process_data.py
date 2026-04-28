import pandas as pd
import numpy as np
import glob

RESISTOR_ERROR = 0.01 # 1% error in the resistance value, which is a common tolerance for resistors.
fileList = [f for f in glob.glob('data/*.csv') if '_processed' not in f]

for fileName in fileList:
    data = pd.read_csv(fileName, usecols=[0,1,2,3], names=['supply_voltage', 'supply_voltage_error', 'diode_voltage', 'diode_voltage_error'], header=None, sep='\t')
    supply_voltage = data['supply_voltage'].to_numpy()
    diode_voltage = data['diode_voltage'].to_numpy()

    # errors from specs on this data sheet: https://www.tek.com/en/datasheet/broad-purpose-digital-multimeters/model-2110-5-1-2-digit-dual-display-digital-multimeter
    # 0.012% of reading + 0.001% of full scale (1 V) for the diode voltage
    diode_voltage_error = (diode_voltage * 0.00012) + (1.0 * 0.00001)

    # the meter auto ranges at 10V
    supply_voltage_range = np.where(supply_voltage <= 10.0, 10.0, 100.0)
    # 0.012% of reading + 0.002% of full scale for the supply voltage
    supply_voltage_error = (supply_voltage * 0.00012) + (supply_voltage_range * 0.00002)

    # match fileName[-8:-5]:
    #     case '001':
    #         resistance = 1e6 # 1 MΩ
    #     case '010':
    #         resistance = 1e4 # 10 kΩ
    #     case '100':
    #         resistance = 1e5 # 100 kΩ

    resistor_voltage = supply_voltage - diode_voltage
    resistor_voltage_error = np.sqrt(supply_voltage_error**2 + diode_voltage_error**2)

    ln_resistor_voltage = np.log(resistor_voltage)
    ln_resistor_voltage_error = resistor_voltage_error / resistor_voltage

    output = pd.DataFrame({
        'diode_voltage': diode_voltage,
        'ln_resistor_voltage': ln_resistor_voltage,
        'ln_resistor_voltage_error': ln_resistor_voltage_error
    })

    fileName = fileName.replace('.csv', '_processed.csv')
    output.to_csv(fileName, index=False, header=False)
