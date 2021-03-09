"""Following the Flask Mega tutorial https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-vii-unit-testing-legacy"""

import os
import unittest

from config import basedir
from app import app, db
from app.models import User

class TestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'test.db')
        self.app = app.test_client()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_make_unique_username(self):
        u = User(username='john', email='john@example.com')
        db.session.add(u)
        db.session.commit()
        username = User.make_unique_username('john')
        assert username != 'john'
        u = User(username=username, email='susan@example.com')
        db.session.add(u)
        db.session.commit()
        username2 = User.make_unique_username('john')
        assert username2 != 'john'
        assert username2 != username

if __name__ == '__main__':
    unittest.main()