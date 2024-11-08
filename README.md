# GloboChallenge3.0
O projeto consiste em um modelo de LLM, que usa progressão linear para analisar textos de possíveis noticias e decidir se são verdadeiros ou falsos.

## Fonte dos dados
Os dados vem de base encontradas na internet, a primeira contem noticias ja classificadas como verdadeiras e falsas, alem de outras.
A segunda base - não pode ser adicionada ao github devido seu tamanho - contem comentarios extraidos do twitter que analisa o sentimento nos comentarios.
A segunda base deve ser renomeada para: `sentimentos.csv`

[base das noticias](https://github.com/jghm-f/FACTCK.BR/blob/master/FACTCKBR.tsv)
[base dos sentimentos](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis?resource=download&select=NoThemeTweets.csv)

## Como executar o projeto
O projeto pincipal é o notbook `verdadeiroFalso.ipynb`, que usa os dados de `df_treino.csv` para treinar o modelo, e o `df_teste.csv` contem alguns poucos dados que podem ser usado para teste.
A base `FACTCKBR.tsv` contem os dados puros, sem tratamento. O notbook `analise.ipynb` é usado para limpar e filtrar os dados.

O projeto ainda contem uma análise sob os sentimentos como um extra, que pode ser localizado no notbook `sentimentos.csv`, a base original não pode ser adicionada ao github. As bases `sentimentos_treino.csv` e `sentimentos_teste.csv`, são usadas para treinamento e teste, respectivamente.
