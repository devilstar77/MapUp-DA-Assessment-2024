from typing import Dict, List

import pandas as pd

#QUESTION NO 1

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
       i = 0
    
    while(i<n):
    
        left = i 
        right = min(i + k - 1, n - 1) 
        while (left < right):
            
            arr[left], arr[right] = arr[right], arr[left]
            left+= 1;
            right-=1
        i+= k
    
# EXAMPLE
arr = [1, 2, 3, 4, 5, 6,7, 8] 

k = 3
n = len(arr) 
reverse(arr, n, k)

for i in range(0, n):
        print(arr[i], end =" ")

    
#QUESTION NO 2

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {} 
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

# input
input1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
output1 = group_by_length(input1)
print(output1)  

input2 = ["one", "two", "three", "four"]
output2 = group_by_length(input2)
print(output2)  
    

#QUESTION NO 3

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    items = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{index}]", sep=sep))
                else:
                    items[f"{new_key}[{index}]"] = item
        else:
            items[new_key] = value
    
    return items

# Example 
nested_input = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_output = flatten_dict(nested_input)
print(flattened_output)

    
 #QUESTION NO 4  

def unique_permutations(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])  
            return
        
        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] not in seen: 
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start] 
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]  

    nums.sort()  
    result = []
    backtrack(0)
    return result

# Example 
input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)

#QUESTION NO 5

import re
from datetime import datetime

def is_valid_date(date_str, date_format):
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False

def find_all_dates(input_string):
    
    patterns = [
        r'(?<!\d)(\d{2})-(\d{2})-(\d{4})(?!\d)', 
        r'(?<!\d)(\d{2})/(\d{2})/(\d{4})(?!\d)',  
        r'(?<!\d)(\d{4})\.(\d{2})\.(\d{2})(?!\d)',  
    ]
    
    found_dates = []
    
   
    for pattern in patterns:
        matches = re.findall(pattern, input_string)
        for match in matches:
            if pattern == patterns[0]:  
                date_str = f"{match[0]}-{match[1]}-{match[2]}"
                if is_valid_date(date_str, "%d-%m-%Y"):
                    found_dates.append(date_str)
            elif pattern == patterns[1]: 
                date_str = f"{match[0]}/{match[1]}/{match[2]}"
                if is_valid_date(date_str, "%m/%d/%Y"):
                    found_dates.append(date_str)
            elif pattern == patterns[2]:  
                date_str = f"{match[0]}.{match[1]}.{match[2]}"
                if is_valid_date(date_str, "%Y.%m.%d"):
                    found_dates.append(date_str)
    
    return found_dates

# sample
input_string = "The events are scheduled on 12-05-2021, 05/12/2021, and 2021.05.12."
print(find_all_dates(input_string))

#QUESTION NO 6

import polyline
import math

def haversine(coord1, coord2):
    R = 6371000  
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c  
    return distance

def decode_polyline_and_calculate_distance(polyline_str):
    
    coordinates = polyline.decode(polyline_str)

    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
   
    distances = [0]  
    
    for i in range(1, len(df)):
       
        prev_point = (df.at[i-1, 'latitude'], df.at[i-1, 'longitude'])
        current_point = (df.at[i, 'latitude'], df.at[i, 'longitude'])
        distance = haversine(prev_point, current_point)
        distances.append(distance)

  
    df['distance'] = distances
    
    return df

polyline_str = "u{_vFz~wq@LwC^vC}D~Hc@r@yC"  
result_df = decode_polyline_and_calculate_distance(polyline_str)
print(result_df)

#QUESTION NO 7

def rotate_and_transform(matrix):
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  

    return final_matrix

#example
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform(matrix)
print(result)


#QUESTION NO 8

from datetime import time

def check_time_completeness(df):
    
    results = pd.Series(index=pd.MultiIndex.from_tuples([], names=["id", "id_2"]), dtype=bool)
    complete_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
    full_day_time_range = {time(hour, minute) for hour in range(24) for minute in range(60)}
    grouped = df.groupby(['id', 'id_2'])

    for (id_val, id_2_val), group in grouped:
        unique_days = set(group['startDay'].unique())
        start_times = pd.to_datetime(group['startTime']).dt.time.unique()
        end_times = pd.to_datetime(group['endTime']).dt.time.unique()
        time_values = set(start_times).union(end_times)
        days_covered = unique_days == complete_days 
        times_covered = time_values == full_day_time_range
        results.loc[(id_val, id_2_val)] = not (days_covered and times_covered)

    return results
ab=pd.read_csv('dataset1.csv')
print(check_time_completeness(ab))
