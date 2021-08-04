 # Trabalho de Conclusão de Curso
##Análise de Sentimentos em Tweets sobre a Pandemia COVID-19 usando Redes Neurais Long Short-Term Memory

Para compilação do código é necessário a intalação das dependências contidas no arquivo **dependencies.sh**.

Para execução do código é preciso adicionar uma pasta de **datas** dentro do diretório **src** e colocar os dados arquivos disponíveis no seguinte link:

[Link dos dados processados](https://drive.google.com/file/d/1_QvOtZKhgJTwuFZEYpilC6IILkueB0Cd/view?usp=sharing)

Para que esses dados sejam reconhecidos existe um arquivo de rotas **import_util**, portanto para qualquer ajuste no caminho ou os dados utilizados deve ser feito alterações nesse arquivo. 

## Nuvem de Adjetivos
Para criar o nuvem de adjetivos foi comentado a lógica de treinamento da main e adicionado a função de geração. A lógica é concentrada em 2 arquivos **"df_util.py"** e **"graphic_util.py"**.
Para a geração da imagem é necessário modificar o caminho de arquivo da função **create_adjetives_cloud**.