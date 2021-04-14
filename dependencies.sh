# pip
sudo apt install python3-pip
pip3 install --upgrade pip

# numpy
pip3 install numpy

# pandas
pip3 install pandas

# scikit-learn
pip3 install scikit-learn

# tensorflow - maybe i will need a setuptools with higher version
pip3 install tensorflow

# gensim
pip3 install gensim==3.6.0

# nltk
pip3 install nltk==3.2.5

pip3 install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html


echo "\n"
echo "The packages has been successfully installed:"
pip3 list


