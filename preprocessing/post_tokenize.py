import sys

STR = '▁'

if __name__ == "__main__":
    ref_fn = sys.argv[1]

    f = open(ref_fn, 'r')

    for ref in f:
        ref = ref.strip()
        input_line = sys.stdin.readline().strip()

        if input_line != "":
            buf = [STR]

            ref_index = 0
            input_index = 0
            while input_index < len(input_line):
                c = input_line[input_index]
                input_index += 1

                if c != ' ':
                    while ref_index < len(ref):
                        c_ = ref[ref_index]
                        ref_index += 1

                        if c_ == ' ':
                            c = STR + c
                        else:  
                            break
                buf += [c]

            sys.stdout.write(''.join(buf) + '\n')
        else:
            sys.stdout.write('\n')
    
    f.close()