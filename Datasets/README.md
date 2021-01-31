== Datasets API ==

Providing some API for specified datasets.

=== IMDB-WIKI ===

API in `imdb_wiki.py`.

- read_mat: Reading the .mat annotation files in IMDB-WIKI datasets and converting them to python objects.
Then store in the cache file (default in `/tmp/imdb_wiki.pth`). It will also return the objects.
The format of the data is `[full_paths_list, face_locations_list, genders_list, ages_list]`
