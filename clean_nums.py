import re
import sys

def main():
    with open(sys.argv[1], 'r') as inp, open("clean_"+sys.argv[2], 'a') as out, open("bad_"+sys.argv[2], 'a') as bad:
        counter = 0
        for line in inp:
            counter += 1
            tokens = line.split("|")[1]
            if ('exp' in tokens) or ('ln' in tokens):
                bad.write(line)
            else:
                out.write(line)
    inp.close()
    out.close()
    bad.close()


if __name__ == '__main__':
    main()