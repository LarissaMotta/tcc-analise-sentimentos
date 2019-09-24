# Rodar script shell? sh dependencies_packages.sh #

# Instalando o pip3 no ambiente
sudo apt-get install python3-pip

# Pacote de processamento de linguagem natural
sudo pip3 install -U spacy
python3 -m spacy download en
python3 -m spacy download pt
