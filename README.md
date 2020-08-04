# ts_test
several tests from ts field

## clonar repositorio

```
git clone https://github.com/griu/ts_test.git
cd ts_test
``` 

## conda envirnement creation

```
conda create -n ts_env python=3.7.3
conda activate ts_env
#python -m pip install statsmodels
#python -m pip freeze > requirements.txt
```

### Para instalar de nuevo el entorno con el requirements

```python -m pip install -r requirements.txt```

### Publicar kernel en pupyterhub:

```
python -m ipykernel install --user --name ts_env --display-name "ts_env"
```

## modificar git ignore y publicar cambios

```
git pull
git add .
git commit -m "Inicializamos entorno"
git push origin master
```

# references
Thanks to IJN-Kasumi 1939–1945 in [Medium](https://medium.com/@sakamoto2000.kim/forecast-arima-gluonts-and-fbprophet-methods-on-the-same-stage-f62d55acc5bb)
and also to Kshif in [github post](https://github.com/zalandoresearch/pytorch-ts).


# open in colab

- With gluonts (CPU): [01_TS_Benchmark.ipynb](http://colab.research.google.com/github/griu/ts_test/blob/master/01_TS_Benchmark.ipynb)
- With pytorchts (GPU): [02_TS_Benchmark_pytorch.ipynb](http://colab.research.google.com/github/griu/ts_test/blob/master/02_TS_Benchmark_pytorch.ipynb)
  - edit -> notebook setting -> Python 3 GPU


