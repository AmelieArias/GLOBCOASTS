## Installation

Clone this repository 

```bash
git clone https://github.com/tomsail/GLOBCOASTS.git
git checkout jrc_version
```

create base environment

```
mamba create -n globcoast python=3.11
mamba activate globcoast
```

(optional: create virtual env):

```
python -mvenv .venv 
source .venv/bin/activate
```


Install the package directly from the source:

```bash
pip install .
```
For development mode (editable install):

```bash
pip install -e .
```

## run Globcoast

```bash
python 150125_CLEANED_GLOBCOASTS.py 
```
