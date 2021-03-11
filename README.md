# Ag-AI
Capstone Project

The goal of this project is to build trust in farmers on the idea of artificial intelligence. We plan to do this by allowing the user to interact with an active learning system, providing information to a machine learning algorithm and seeing how their input changes the accuracy of the algorithm.

The CapstoneMain is used to run this project. Images folder is used to hold all images. Images must be separated by classification by adding a folder that holds all images of the classification. There must be a path to each file in master to run this program.

The user will interact with a web-based user interface, where they’ll be asked to label pictures of corn as either healthy or unhealthy. 

HOW TO USE:
Install Steps
Assuming using Visual Studio Code or similar, have python 3 installed and setup and on windows
While in project folder open the terminal and run 
	python -m venv env
Install required software
python -m pip install --upgrade pip
python -m pip install flask
python -m pip install Flask-WTF
python -m pip install Flask-bootstrap
python -m pip install sklearn
python -m pip install pandas
python -m pip install boto3
python -m pip install fsspec
python -m pip install s3fs
At this point you might need to reinstall boto3 in s3fs uninstalled it

For creation of the database you’ll need 2 more pip installs
	python -m pip install flask-sqlalchemy
	python -m pip install flask-migrate
	(flask-migrate is used for updating the structure of the database incase revisions are necessary)
You'll also need to install Flask-login and email-validator
	python -m pip install flask-login
	python -m pip install email-validator
  
Now that everything is installed run command
$env:FLASK_APP = "flask_app"
If running from cmd line instead of IDE use 
	set FLASK_APP=flask_app
This changes the name of the flask apps default program to flask_app 
Now to run the app use
	python -m flask run
This should output something like
* Serving Flask app "flask_app"
 * Environment: production
 	  WARNING: This is a development server. Do not use it in a production deployment.
 	  Use a production WSGI server instead.
 	* Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
At this point copy and paste the http: line into your favourite internet browser and it should be running the site.


!!!DO NOT FOLLOW THIS SECTION ALREADY GENERATED CODE WILL BE UPLOADED!!!
To create the database (using code from github) run these commands (SPECIFICALLY FOR CREATING USER TABLE) 
	python -m flask db init
This should output something like 
Creating directory C: … done
Creating directory C: ...  done
Generating C: ...  done
Generating C: ...  done
Generating C: ...  done
Generating C: ...  done
Please edit configuration/connection/logging settings in 'C:’ before preceding 

python -m flask db migrate -m "users table"
This should output
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.autogenerate.compare] Detected added table 'user'
INFO  [alembic.autogenerate.compare] Detected added index 'ix_user_email' on '['email']'
INFO  [alembic.autogenerate.compare] Detected added index 'ix_user_nickname' on '['nickname']'
Generating C: …\migrations\versions\79cc6c580ecc_users_table.py ...  done
This merged the database with a new one “users table” (doesn’t exist)
NOTE “users table” is the name of the database
To update the database you’ll need need to use “python -m flask db upgrade” not migrate


python -m flask db upgrade


