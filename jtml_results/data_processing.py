import csv
import math

# Define a function to calculate the RMSD between two lists of values
def calculate_rmsd(data1, data2):
    n = len(data1)
    sum_of_squared_differences = sum((x - y) ** 2 for x, y in zip(data1, data2))
    rmsd = math.sqrt(sum_of_squared_differences / n)
    return rmsd

# Initialize dictionaries to store data for each method
method1_data = { 'x_translation': [], 'y_translation': [], 'z_translation': [], 'x_rotation': [], 'y_rotation': [], 'z_rotation': [] }
method2_data = { 'x_translation': [], 'y_translation': [], 'z_translation': [], 'x_rotation': [], 'y_rotation': [], 'z_rotation': [] }

# Read data from method1.csv
with open('jtml_results/hum.jts', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        for i, measurement in enumerate(row[2:]):
            method1_data[list(method1_data.keys())[i]].append(float(measurement))

# Read data from method2.csv
with open('jtml_results/JTML_HUM_KIN.jtak', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        for i, measurement in enumerate(row[2:]):
            method2_data[list(method2_data.keys())[i]].append(float(measurement))

# Calculate RMSD for each measurement
rmsd_values = {}

for measurement in method1_data.keys():
    rmsd_values[measurement] = calculate_rmsd(method1_data[measurement], method2_data[measurement])

# Print the RMSD values for each measurement
print("RMSD values between Method 1 and Method 2:")
for measurement, rmsd in rmsd_values.items():
    print(f"{measurement}: {rmsd}")