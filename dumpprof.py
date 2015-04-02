import pstats
pstats.Stats('prof').strip_dirs().sort_stats("cumulative").print_stats()
