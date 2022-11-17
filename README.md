# CeNTREX-transition-finder
 
Docker file to run the transition finder.
Build the image with:
```
docker build --progress plain --no-cache -t centrex_transition_finder .
```

and run with
```
docker run -d -p 8501:8501 --name centrex-transition-finder centrex_transition_finder
```
which will create a container with the name centrex-transition-finder and exposes port 8051 for the streamlit interface.