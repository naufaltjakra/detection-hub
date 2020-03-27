# Project Overview

> This is a Machine Learning Web App Project using
> Random Forest Algorithm to detect Phishing Website.
> The dataset contains 17 attributes.

# Run Instruction (Using Windows 10)
**1. Install All Dependencies**

On console..
```sh
PC/user/you> pip install -r requirements.txt
```

**3. Setup the Flask Environment**

On CMD..
```sh
PC/user/you> set FLASK_APP=app
```

On PowerShell..
```sh
PC/user/you> $env:FLASK_APP = "main.py"
PC/user/you> $env:FLASK_DEBUG = 1 (This is optional)
```

Or visit env [documentation](https://flask.palletsprojects.com/en/1.0.x/cli/)

**4. Run Flask**

On Console..
```sh
PC/user/loc> flask run
```

**5. Open Browser**

- On url => localhost:5000 (usually)