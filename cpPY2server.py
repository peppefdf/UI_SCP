import os

for file in os.listdir('./'):
    # check only text files
    if file.endswith('.py'):
        print(file)

name = input("Enter filename: ")

os.system('scp ' + name + ' cslgipuzkoa@82.116.171.111:/home/cslgipuzkoa/')

