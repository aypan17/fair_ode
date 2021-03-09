import re
import sys
import matplotlib.pyplot as plt

def main():
    with open(sys.argv[1], 'r') as inp:
        nums = [[] for _ in range(2, 16)]
        overflow = 0
        unk = 0
        for line in inp:
            lst = line.split('INT')
            for n in lst:
                if len(n) > 15:
                    overflow += 1
                    print(n)
                elif n[-1] == 'I':
                    try:
                        nums[len(n)-1-2].append(int(n[:-1]))
                    except ValueError:
                        unk += 1
                else:
                    try:
                        nums[len(n)-2].append(int(n))
                    except ValueError:
                        unk += 1
        '''
        for k in range(2, 10):
            plt.hist(nums[k-2], bins=30)
            plt.xlabel("Value")
            plt.ylabel("Count")
            plt.savefig(sys.argv[1]+'_dist_'+str(k-1)+'.png')
            if k % 2 == 1:
                plt.clf()
        '''
        print(overflow, unk)
    inp.close()
    


if __name__ == '__main__':
    main()
