import os
import csv
import random

def split_csv(input_file, output_dir, percentages):
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        
        # Assuming the first row is the header
        #header = next(reader)  
        
        rows = list(reader)
        total_rows = len(rows)
        
        if sum(percentages) != 100:
            raise ValueError("Sum of percentages should be 100")
        
        output_files = []
        for i, percent in enumerate(percentages):
            output_file = os.path.join(output_dir, f"output_{i+1}.csv")
            output_files.append((output_file, percent))
            
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                # writer.writerow(header)
                
                num_rows = int(total_rows * percent / 100)
                selected_rows = random.sample(rows, num_rows)
                writer.writerows(selected_rows)
                
    return output_files

if __name__ == "__main__":
    input_file = "C:/MyLibrary/Data/celebA_Anno/list_attr_celeba.csv"  # Replace with your input CSV file
    output_directory = "C:/MyLibrary/Data/celebA_Anno/list_attr"  # Replace with your desired output directory
    
    # List of percentages for splitting (should sum up to 100)
    percentages = [80, 10, 10]  # Adjust the percentages as needed
    
    try:
        os.makedirs(output_directory, exist_ok=True)
        result = split_csv(input_file, output_directory, percentages)
        print("CSV file split completed:")
        for file, percent in result:
            print(f"{file}: {percent}%")
    except Exception as e:
        print("An error occurred:", e)