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
python -m pip install fbprophet
python -m pip install --upgrade mxnet==1.6 gluonts
python -m pip install torch torchvision
python -m pip install pytorchts
python -m pip install plotly dash
python -m pip freeze > requirements.txt
```

### Para instalar de nuevo el entorno con el requirements

```python -m pip install -r requirements.txt```

### Publicar kernel en pupyterhub:

```python -m pip install --upgrade ipykernel```

## modificar git ignore y publicar cambios

```
git pull
git add .
```
