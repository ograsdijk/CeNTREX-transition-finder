# app/Dockerfile

FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ograsdijk/CeNTREX-TlF-Hamiltonian.git centrex_hamiltonian

RUN pip3 install ./centrex_hamiltonian

RUN git clone https://github.com/ograsdijk/CeNTREX-TlF-Couplings.git centrex_couplings

RUN pip3 install ./centrex_couplings

RUN git clone https://github.com/ograsdijk/CeNTREX-transition-finder.git transition_finder

RUN pip3 install -r ./transition_finder/requirements.txt

ENTRYPOINT ["streamlit", "run", "./transition_finder/transition_finder.py", "--server.port=8501", "--server.address=0.0.0.0"]