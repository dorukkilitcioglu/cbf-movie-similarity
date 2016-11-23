# cbf-movie-similarity
Content-based similar movie recommendation engine.

We try to implement and improve the paper by Fleischman et. al.<sup>[1]</sup> 

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

1: Michael Fleischman and Eduard Hovy. 2003. Recommendations without user preferences: a natural language processing approach. In Proceedings of the 8th international conference on Intelligent user interfaces (IUI '03). ACM, New York, NY, USA, 242-244.  
DOI = http://dx.doi.org/10.1145/604045.604087  
URL = https://www.cs.cmu.edu/~hovy/papers/03IUI-recommender.pdf
