import sys
import io
import json
import matplotlib.pyplot as plt

def main():
    inp = sys.argv[1]
    acc = {}
    with io.open(inp, mode='r', encoding='utf-8') as f1:
        for line in f1:
            if "__log__:" in line:
                d = json.loads(line.split("__log__:", 1)[1])
                acc[d['epoch']] = d['valid_prim_fwd_acc']
    
    accs = []
    for i in range(len(acc)):
        accs.append(acc[i])
    #plt.plot(accs, range(len(accs)))
    print(accs)

if __name__ == '__main__':
    main()