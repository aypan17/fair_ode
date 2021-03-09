import re
import sys

def main():
    with open(sys.argv[1], 'r') as inp, open(sys.argv[2], 'a') as out:
        for line in inp:
            out.write(re.sub("[^0-9INT+-]", "", line))
    inp.close()
    out.close()


if __name__ == '__main__':
    main()