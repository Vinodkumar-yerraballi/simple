create env

```bash

conda create -n own python=3.9

```

activate env

```bash
conda activate own

```

created requirements.txt

```bash

pip  install -r requirements.txt
```

```bash
git init
```

```bash
dvc init
```

```bash
dvc add data_given/winequality-red.csv
```

```bash
git add . && git commit -m "update my READNE.md"

```

```bash
git remote add origin https://github.com/Vinodkumar-yerraballi/simple.git
git branch -M main
git push origin main
```

tox command

```bash
tox
```

tox rebuliding

```bash
tox -r
```

pytest command

```bash
pytest -v
```

setup commands -

```bash
pip install -e .
```

build your own package commands-

```bash
python setup.py sdist bdist_wheel
```
