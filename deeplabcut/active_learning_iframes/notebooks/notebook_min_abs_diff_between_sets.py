# From https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/


# Python3 program for the
# above approach
# A Recursive C program to
# solve minimum sum partition
# problem.

# Function to find the minimum sum


def findMinRec(i,
               list_S_total, 
               list_S1):
 
    sumTotal = sum(list_S_total)
    sumS1 = sum(list_S1)
    # If we have reached last element.
    # Sum of one subset is sumCalculated,
    # sum of other subset is sumTotal-
    # sumCalculated.  Return absolute
    # difference of two sums.
    if (i == 0):
        return (abs((sumTotal - sumS1) - sumS1), list_S1)
 
    # For every item arr[i], we have two choices
    # (1) We do not include it first set
    # (2) We include it in first set
    # We return minimum of two choices
    list_S1_added = list_S1.copy()
    list_S1_added.append(list_S_total[i - 1])
    return min(findMinRec(i-1,
                          list_S_total, 
                          list_S1_added),
               findMinRec(i-1,
                          list_S_total, 
                          list_S1))
 
# Returns minimum possible
# difference between sums
# of two subsets
 
 
def findMin(list_S_total):

    # Compute initial sum in set S1
    list_S1_ini = []
     
    # Compute result using
    # recursive function
    return findMinRec(len(list_S_total), # i
                      list_S_total, 
                      list_S1_ini)

######################################################
# Driver code
if __name__ == "__main__":
    dict_n_files_per_dir_per_shuffle = dict()
    # dict_n_files_per_dir_per_shuffle[0] = [1,6,11,5] # test
    dict_n_files_per_dir_per_shuffle[1] = {'BrownHorseinShadow': 308,
                                            'BrownHorseintoshadow': 289,
                                            'Brownhorselight': 306,
                                            'Brownhorseoutofshadow': 341,
                                            'Chestnuthorseongrass': 376,
                                            'Sample1': 174,
                                            'Sample10':235,
                                            'Sample11': 256,
                                            'Sample14': 168,
                                            'Sample15': 154,
                                            'Sample16': 212,
                                            'Sample18': 159,
                                            'Sample19': 134,
                                            'Sample2': 330,
                                            'Sample20': 180,
                                            'Sample3': 342,
                                            'Sample4': 305,
                                            'Sample6': 376,
                                            'Sample9': 359,
                                            'TwoHorsesinvideobothmoving': 181}
                          
    dict_n_files_per_dir_per_shuffle[2] = {'BrownHorseinShadow': 308,
                                            'BrownHorseintoshadow': 289,
                                            'GreyHorseNoShadowBadLight': 286,
                                            'Sample1': 174,
                                            'Sample10': 235,
                                            'Sample12': 288,
                                            'Sample14': 168,
                                            'Sample15': 154,
                                            'Sample16': 212,
                                            'Sample17': 240,
                                            'Sample19': 134,
                                            'Sample2': 330,
                                            'Sample20': 180,
                                            'Sample3': 342,
                                            'Sample4': 305,
                                            'Sample5': 295,
                                            'Sample6': 376,
                                            'Sample7': 262,
                                            'Sample9': 359,
                                            'TwoHorsesinvideobothmoving': 181}
                                           
    dict_n_files_per_dir_per_shuffle[3] = {'BrownHorseintoshadow': 289,
                                            'Brownhorselight': 306,
                                            'ChestnutHorseLight': 318,
                                            'GreyHorseLightandShadow': 356,
                                            'Sample1': 174,
                                            'Sample10': 235,
                                            'Sample11': 256,
                                            'Sample14': 168,
                                            'Sample15': 154,
                                            'Sample16': 212,
                                            'Sample17': 240,
                                            'Sample18': 159,
                                            'Sample2': 330,
                                            'Sample20': 180,
                                            'Sample3': 342,
                                            'Sample5': 295,
                                            'Sample6': 376,
                                            'Sample7': 262,
                                            'Sample8': 388,
                                            'Sample9': 359}
    for sh in range(1,len(dict_n_files_per_dir_per_shuffle.keys())+1):      


        min_diff, list_S1 = findMin(list(dict_n_files_per_dir_per_shuffle[sh].values()))
        list_S2 = [el for el in dict_n_files_per_dir_per_shuffle[sh].values() 
                      if el not in list_S1]
        
        print("Minimum difference between the two sets for shuffle {}: {}".format(sh, min_diff))
        print('Elements in S1: {}'.format(list_S1))
        # print('Elements in S1: {}'.format([dict_n_files_per_dir_per_shuffle[sh][el] for el in list_S1]))
        print('Elements in S2: {}'.format(list_S2))
        # print(S1_final)
        print('---------------------------')

