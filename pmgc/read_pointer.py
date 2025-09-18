import pdb
import itertools


class PrintArray(gdb.Command):
    """Print values of an array given a pointer and a length.
    Usage: print_array <ptr_expr> <length>
    Example: print_array arr 5
    """

    def __init__(self):
        super().__init__("print_array", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) < 2:
            print("Usage: print_array <ptr_expr> <length> ...")
            return
        

        ptr_expr =args[0]

        length_str = args[1:]
        

        try: 
            lengths = [int(length) for length in length_str];
            max_indices = [int(length) - 1 for length in length_str]  
            # maximum value for each index
        except:
            print("Usage: print_array <ptr_expr> <length> ...")
            print("Issue when converting length to numbers")



        try:
            ptr = gdb.parse_and_eval(ptr_expr)   # gdb.Value (pointer)
        except gdb.error as e:
            print(f"Error evaluating pointer: {e}")
            return
        
        for indices in itertools.product(*[range(length) for length in lengths]):

            #Flatten multi-dimensional indices to single index
            flat_index = indices[-1]



            reverse_lengths = tuple(reversed(lengths))

            for idx, stride in enumerate(reverse_lengths[1:], start=2):
                flat_index *= stride
                flat_index = flat_index + indices[-idx]

            try:
                
                val = ptr[flat_index]
                print(val, end=' ')

                # find the first differing index 
                diff_index = next((i for i, (a, b) in 
                                   enumerate(zip(indices[::-1], max_indices)) if a != b), 2)



                #print as many newlines as count_may
                print("\n"* diff_index, end='')
            
            except gdb.error as e:
                
                print(f"Error evaluating pointer {e}")
                return    

PrintArray()

