#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Assignment No 1
# Machine Leaning
# Submitted By - SP23-RCS-022


# In[13]:


## Task 0: Python Installation

### 0.1 Anaconda is an open-source distribution of Python programming language for scientific computing. Installing 
### Anaconda will automatically install Python and the necessary packages for development and analysis. Go to the official 
### Anaconda page. Download the installer compatible with your system to install a fresh copy of Anaconda. Run the 
### installer and follow the instructions there.
### 0.2 Verifying Installation
### To test your installation, open Command Prompt or Anaconda Prompt and run the command
###conda list
### For a successful installation, a list of installed packages appears.

####The anaconda is successfully intalled. On run the command 'conda list' on anaconda promt, a list of installed/
###pakages appears.


# In[ ]:


## Task 1: Lists, Dictionaries, Tuples


# In[1]:


###Task 1.1: Creaye a list: nums = [3, 5, 7, 8, 12], make another list named ‘cubes’ and append the cubes of the given list ‘nums’ 
###in this list and print it.

nums = [3, 5, 7, 8, 12]
cubes = []

for num in nums:
    cubes.append(num ** 3)
    
print(cubes)


# In[14]:


###Task 1.2: Create an empty dictionary: dict = {}, add the following data to the dictionary: ‘parrot’: 2, ‘goat’: 4, ‘spider’: 8, ‘crab’: 
###10 as key value pairs.

dictLegs = {}

dictLegs['parrot'] = 2
dictLegs['goat'] = 4
dictLegs['spider'] = 8
dictLegs['crab'] = 10

###Task 1.3: Use the ‘items’ method to loop over the dictionary (dict) and print the animals and their corresponding legs.Sum 
###the legs of each animal, and print the total at the end.

totalLegs = 0

for animal, legs in dictLegs.items():
    print(f"{animal}: {legs} legs")
    totalLegs += legs
    
print(f"Total legs: {totalLegs}")


# In[5]:


###Task 1.4: Create a tuple: A = (3, 9, 4, [5, 6]), change the value in the list from ‘5’ to ‘8’

A = (3, 9, 4, [5, 6])
A[3][0] = 8  # Modify value 5 to value 8 in the list inside the tuple

print(A)

###Task 1.5: Delete the tuple A.

del A # A is now deleted.


# In[15]:


###Task 1.6: Create another tuple: B = (‘a’, ‘p’, ‘p’, ‘l’, ‘e’), print the number of occurrences of ‘p’ in the tuple.

B = ('a', 'p', 'p', 'l', 'e')

countP = B.count('p')
print(f"Occurrences of 'p': {countP}")

###Task 1.7: Print the index of ‘l’ in the tuple.

indexL = B.index('l')
print(f"Index of 'l': {indexL}")


# In[ ]:


#Task 2: Numpy


# In[29]:


###Task 2.1: Convert matrix A into a NumPy array

import numpy as np

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

###Task 2.2: Use slicing to pull out the subarray consisting of the first 2 rows and columns 1 and 2. Store it in b which is a numpy 
###array of shape (2, 2).

b = A[0:2, 1:3]  # First 2 rows and columns 1 and 2
print(b)

###

C = np.empty_like(A)

### Task 2.4: Add the vector z to each column of the matrix ‘A’ with an explicit loop and store it in 'C'.

z = np.array([1, 0, 1])
for i in range(A.shape[0]):
     C[:, i] = A[:, i] + z
    
print(C)


# In[31]:


###Task 2.5: Add and print the matrices X and Y.

X = np.matrix([[1, 2],
              [3, 4]])
Y = np.matrix([[5, 6],
              [7, 8]])
v = np.array([9, 10])

sumXY = X + Y
print(sumXY)

###Task 2.6: Multiply and print the matrices X and Y.

productXY = X @ Y  # Matrix multiplication
print(productXY)

###Task 2.7: Compute and print the element wise square root of matrix Y.

sqrtY = np.sqrt(Y)
print(sqrtY)

###Task 2.8:  Compute and print the dot product of the matrix X and vector v

dotProductXv = np.dot(X,v)
print(dotProductXv)

###Task 2.9: Compute and print the sum of each column of X.

columnSumsX = np.sum(X, axis=0)
print(columnSumsX)


# In[ ]:


#Task 3: Functions and Loops


# In[33]:


###Task 3.1: Create a function ‘Compute’ that takes two arguments, distance and time, and use it to calculate velocity.


def compute(distance, time):
    if time == 0:
        return "Time cannot be zero."
    velocity = distance / time
    return velocity

###Task 3.2: Make a list named ‘even_num’ that contains all even numbers up till 12. Create a function ‘mult’ that takes the list
###‘even_num’ as an argument and calculates the products of all entries using a for loop.

even_num = [2,4,6,8,10,12]  # List of even numbers till 12

def mult(numbers):
    product = 1
    for num in numbers:
        product *= num
    return product

# Example usage
result = mult(even_num)
print(f"Product of even numbers: {result}")


# In[ ]:


# Task 4: Pandas


# In[43]:


import pandas as pd

### Creating the DataFrame
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

df = pd.DataFrame(data)

### Task 4.1: Print only the first two rows of the dataframe.
print(df.head(2))

### Task 4.2: Print the second column.
print(df['C2'])

### Task 4.3: Change the name of the third column from 'C3' to 'B3'.
df.rename(columns={'C3': 'B3'}, inplace=True)

### Task 4.4: Add a new column to the dataframe and name it 'Sum'.
df['Sum'] = 0  # Initialize the new column

### Task 4.5: Sum the entries of each row and add the result in the column 'Sum'.
df['Sum'] = df.sum(axis=1)

### Print the updated DataFrame
print("Updated DataFrame:")
print(df)


# In[54]:


import pandas as pd

### Task 4.6: Read CSV file named ‘hello_sample.csv’ into a Pandas dataframe
csv_df = pd.read_csv('hello_sample.csv')

### Task 4.7: Print complete dataframe
print("\n")  # Empty line before
print(csv_df)
print("\n")  # Empty line after

### Task 4.8: Print only bottom 2 records of the dataframe
print("\n")  # Empty line before
print(csv_df.tail(2))
print("\n")  # Empty line after

### Task 4.9: Print information about the dataframe
print("\n")  # Empty line before
print(csv_df.info())
print("\n")  # Empty line after

### Task 4.10: Print shape (rows x columns) of the dataframe
print("\n")  # Empty line before
print(csv_df.shape)
print("\n")  # Empty line after

### Task 4.11: Sort the data of the DataFrame using column ‘Weight’
### Assuming 'Weight' is a column in your DataFrame
print("\n")  # Empty line before
if 'Weight' in csv_df.columns:
    sorted_df = csv_df.sort_values(by='Weight')
    print(sorted_df)
else:
    print("\n'Weight' column not found in DataFrame.")
print("\n")  # Empty line after

### Task 4.12: Use isnull() and dropna() methods of the Pandas dataframe
print("\n")  # Empty line before
null_values = csv_df.isnull().sum() 
print("Null values in each column:\n", null_values)
print("\n")  # Empty line after

### Drop rows with any null values
cleaned_df = csv_df.dropna()
print("DataFrame after dropping null values:\n", cleaned_df)
print("\n")  # Empty line after


# In[ ]:




