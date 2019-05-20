from process_data.bios_to_table import bios_to_table


# import sys

fakeDataset = True

datasatPath = '../fake_dataset/' if fakeDataset else '../dataset/' 

bios_to_table(datasatPath)

# # de 
# a = {
# 0: 'artist0',
# 1: 'artist1',
# 2: 'artist2',
# 3: 'artist3',
# 4: 'artist4'
# }

# print(list(a.values()))
# print(list(a.values())[2])



