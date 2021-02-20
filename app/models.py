
from app import db, login 
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

"""
Setup the basic layout for the database's users inheirting the base class for models from Flask-SQLAlchemy (inherited via db.Model)
    'id' stores the primary key for the user. Each user will be assigned a unique id
    'username' stores the users chosen nickname or username. Unique
    'email' store the users email. Unique
    'password_hash' a hashed item that represents a user password (will not directly store the password)
"""
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    Confidence_Storage = db.relationship('Confidence', backref='creator', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)



"""
Setup the basic layout for the user's csvs (storing tehir confidence rate) inheirting the base class for models from Flask-SQLAlchemy (inherited via db.Model)
    'id' stores the primary key for the model. Each csv will be assigned a unique id
    'user_id' is a foreign key that connects the confidence data to the user
    'data' is a LargeBinary file that will be the csv file. Should instead create a file system and store csv's there and change 'data' to instead be a path to the file
""" 
class Confidence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    data = db.Column(db.LargeBinary)


def __repr__(self):
    return '<User {}>'.format(self.username)

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

