Imports and Installation:
- The current imports are argparse, json, math, random, from collections --> Counter, from pathlib --> Path and from typing --> Dict, Iterable, List, Sequence, and Tuple.

Running:
- Running the ngram.py file should be the entire training or procedural process however the program does rely on preemptivley cloning, parsing, and tokenizing all of the methods and having it into a file. If it was added putting it into the methods.txt file would be required to function properly. As well the second test set is found in the test.txt file which is necessary to get both json outputs. Otherwise all paths should be generalized in the program and it should be fine to run the ngram.py from anywhere. This includes the output locations it should work to just run the program from anywhere and it will put the json in the folders with the rest of the files. 

Parameters:
- The most important parameter is n for the context window which is currently set to 3, 5, and 7 each individually and seperately run. there is also an alpha parameter for the smoothing that is currently set to 0.1. 