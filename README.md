# Exploring witch stories based on their locality ([hosted here](https://brutenis.net/wossidia/))

This repository contains the code for a study of the [ISEBEL](http://search.isebel.eu/)/[Wossidia](https://apps.wossidia.de/webapp/run) datasets based on their location. These datasets include various stories about witches, werewolves and various other legends/tales as a graph-database. Here we create a website which analyzes that graph-database, extracts the places, assigns it various keywords, clusters it based on those keywords and then plots it on an interactive website using plotly.

## Installation

* Install requirements:

```
pip install -r requirements.txt
```

* Run the server:

```
python server.py
```

* Go to `localhost:8050`

## Run with Apache2 mod_wsgi

The `wsgi.py` file is meant to be run with mod_wsgi.

* Install mod_wsgi:

```
apt install libapache2-mod-wsgi-py3
```

* Add this line in the apache site .conf file:

```
WSGIDaemonProcess wsgi.py threads=3 display-name=%{GROUP} python-home=<venv-path>
```