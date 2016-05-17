# cbf-movie-similarity
Content-based similar movie recommendation engine.

We try to implement and improve https://www.cs.cmu.edu/~hovy/papers/03IUI-recommender.pdf

In order to create the necessary data file, download and unzip the sample list files from the IMDB:  
`ftp://ftp.fu-berlin.de/pub/misc/movies/database/`.  
The necessary files are:
```
genres.list
language.list
movies.list
plot.list
actors.list
actresses.list
```
Simply run through the `Data_Prepare` notebook to create the data file. Other notebooks will use the generated data file.
