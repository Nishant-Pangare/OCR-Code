# import pandas as pd

# # Load the extracted data from the CSV file without headers
# extracted_csv_path = "Final OCR Code/results11_trial.csv"
# df = pd.read_csv(extracted_csv_path, header=None)

# # Add appropriate headers to the DataFrame
# df.columns = ['DATE', 'MEMO NO', 'VEHICLE', 'FUEL TYPE', 'RATE', 'QUANTITY', 'SALE/OTH', 'BALANCE']

# # Define the headers for the new DataFrame
# headers = ['DATE', 'MEMO NO', 'VEHICLE', 'PARTICULARS', 'NET DEBIT', 'CSH/EXP', 'SALE/OTH', 'CREDIT', 'BALANCE']

# # Create a new DataFrame with the desired structure
# new_df = pd.DataFrame(columns=headers)

# # Set the DATE and BALANCE columns
# new_df['DATE'] = df['DATE']
# new_df['BALANCE'] = df['BALANCE']

# # Set the first row values for MEMO NO, VEHICLE, PARTICULARS, and SALE/OTH
# new_df.at[0, 'MEMO NO'] = df.at[0, 'MEMO NO']
# new_df.at[0, 'VEHICLE'] = df.at[0, 'VEHICLE']
# new_df.at[0, 'PARTICULARS'] = 'OPENING BALANCE'
# new_df.at[0, 'SALE/OTH'] = df.at[0, 'BALANCE']

# # Fill the MEMO NO and VEHICLE columns from the second row onwards
# new_df.loc[1:, 'MEMO NO'] = df.loc[1:, 'MEMO NO']
# new_df.loc[1:, 'VEHICLE'] = df.loc[1:, 'VEHICLE']

# # Combine FUEL TYPE, RATE, and QUANTITY into the PARTICULARS column from the second row onwards
# new_df.loc[1:, 'PARTICULARS'] = df.loc[1:, 'FUEL TYPE'] + ' ' + df.loc[1:, 'RATE'].astype(str) + ' ' + df.loc[1:, 'QUANTITY'].astype(str)

# # Fill the SALE/OTH column with the corresponding values from the second row onwards
# new_df.loc[1:, 'SALE/OTH'] = df.loc[1:, 'SALE/OTH']

# # Save the new DataFrame to a CSV file
# output_csv_path = "Final OCR Code/final_results.csv"
# new_df.to_csv(output_csv_path, index=False)

# # Print the new DataFrame
# print(new_df)

import pandas as pd

# Load the extracted data from the CSV file without headers
extracted_csv_path = "Final OCR Code/results11_trial.csv"
df = pd.read_csv(extracted_csv_path, header=None)

# Add appropriate headers to the DataFrame
df.columns = ['DATE', 'MEMO NO', 'VEHICLE', 'FUEL TYPE', 'RATE', 'QUANTITY', 'SALE/OTH', 'BALANCE']


# Concatenate 'FUEL TYPE', 'RATE', and 'QUANTITY' columns into 'PARTICULARS' column
df['PARTICULARS'] = df['FUEL TYPE'] + '     ' + df['RATE'].astype(str) + '  ' + df['QUANTITY'].astype(str)


# Shift the values in the 'MEMO NO' column one place ahead
df['MEMO NO'] = df['MEMO NO'].shift(1)
df['VEHICLE'] = df['VEHICLE'].shift(1)
df['PARTICULARS'] = df['PARTICULARS'].shift(1)

# Set the first cell of 'PARTICULARS' column to "OPENING BALANCE"
df.at[0, 'PARTICULARS'] = "OPENING BALANCE"

# Set the first cell of 'MEMO NO' column to NaN
df.at[0, 'MEMO NO'] = float('NaN')
df.at[0, 'VEHICLE'] = float('NaN')

# Drop the original columns 'FUEL TYPE', 'RATE', and 'QUANTITY'
df.drop(columns=['FUEL TYPE', 'RATE', 'QUANTITY'], inplace=True)

df['SALE/OTH'] = df['SALE/OTH'].shift(1)
df.at[0,'SALE/OTH'] = df.at[0, 'BALANCE']

# Reorder the columns
df = df[['DATE', 'MEMO NO', 'VEHICLE', 'PARTICULARS', 'SALE/OTH', 'BALANCE']]

# Print the DataFrame with headers
print(df)

df['SALE/OTH'].astype('float')

# Save the DataFrame with headers to a new CSV file (optional)
output_csv_with_headers_path = "Final OCR Code/final_results.csv"
df.to_csv(output_csv_with_headers_path, index=False)

# Print the path to the new CSV file
print(f"CSV file with headers saved to: {output_csv_with_headers_path}")
