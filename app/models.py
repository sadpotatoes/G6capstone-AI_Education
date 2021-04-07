from app import db, login 
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

"""
Setup the basic layout for the database's users inheirting the base class for models from Flask-SQLAlchemy (inherited via db.Model)
    'id' stores the primary key for the user. Each user will be assigned a unique id
    'username' stores the users chosen nickname or username. Unique
    'email' store the users email. Unique
    'password_hash' a hashed item that represents a user password (will not directly store the password)
    'Confidence_storage' is a relationship between tables Confidence and User
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

    @staticmethod
    def make_unique_username(username):
        if User.query.filter_by(username=username).first() is None:
            return username
        version = 2
        while True:
            new_username = username + str(version)
            if User.query.filter_by(username=new_username).first() is None:
                break
            version += 1
        return new_username


"""
Setup the basic layout for the user's csvs (storing their confidence rate) inheirting the base class for models from Flask-SQLAlchemy (inherited via db.Model)
    'id' stores the primary key for the model. Each csv will be assigned a unique id
    'user_id' is a foreign key that connects the confidence data to the user via the users id (not username)
    'img_names' is a string that that holds the names of all images the user selected
    'img_labels' is a string that that holds the labels for each image
    'previous' is a string that hold the number of images at each instance accuracy is generated so users can see their previous rates
    'accuracy_rate' is a float that holds the users most recent accuracy rate, for use in the leaderboards
""" 
class Confidence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    img_names = db.Column(db.String)
    img_labels = db.Column(db.String)
    """Since img_names stores the images in the orders they are selected we can regenerate previous accuracy rates using just the number of images in the string
    For instance lets say someone generates their accuracy immediately after their first 10 selection then says 6 images are wrong in feed back
    if we store '10,16' then we know when each accuracy was generated and can regen them based off that information
    This way we don't need to store every previous instance of accuracy and their images"""
    previous = db.Column(db.String)
    accuracy_rate = db.Column(db.Float)


def __repr__(self):
    return '<User {}>'.format(self.username)

@login.user_loader
def load_user(id):
    return User.query.get(int(id))
