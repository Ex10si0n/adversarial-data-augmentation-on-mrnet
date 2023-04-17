import os
import pandas as pd

# Define the directory where the CSV files are located
csv_dir = './'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]

# Create a Pandas Excel writer object
excel_writer = pd.ExcelWriter('combined_data.xlsx', engine='xlsxwriter')

# Loop through the CSV files
for csv_file in csv_files:
    # Read in the CSV file
    csv_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    # Get the sheet name (without "combined_")
    sheet_name = csv_file.replace('combined_', '').replace('.csv', '')
    
    # Write the data to a sheet in the Excel file
    df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

# Save the Excel file
excel_writer.save()

