# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
"""
from flask import Flask
from flask import render_template, flash, redirect, url_for, session, request, jsonify
from app import app, db
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm, LoginForm
from flask_bootstrap import Bootstrap
from flask_login import logout_user, current_user, login_user
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
import boto3
from io import StringIO
from app.models import User, Confidence
from app.forms import RegistrationForm

bootstrap = Bootstrap(app)

def getData():
    """
    Gets and returns the csvOut.csv as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    s3 = boto3.client('s3')
    path = 's3://cornimagesbucket/csvOut.csv'

    data = pd.read_csv(path, index_col = 0, header = None)
    data.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']

    data_mod = data.astype({'8': 'int32','9': 'int32','10': 'int32','12': 'int32','14': 'int32'})
    return data_mod.iloc[:, :-1]

def createMLModel(data):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the features for each image

    Returns
    -------
    ml_model : ML_Model class object
        ml_model created from the training set.
    train_img_names : String
        The names of the images.
    """
    train_img_names, train_img_label = list(zip(*session['train']))
    """
    if current_user.is_authenticated:
        user = User.query.filter_by(username = current_user.username).first()
        if Confidence.query.filter_by(user_id = user.id).first():
            healthy_string = Confidence.query.filter_by(user_id = user.id).first().healthy_data
            healthy_list = healthy_string.split(',')
            train_img_label = []
            train_img_names = []
            for i in healthy_list:
                if i:
                    train_img_names.append(i)
                    train_img_label.append('H')

            blighted_string = Confidence.query.filter_by(user_id = user.id).first().blighted_data
            blighted_list = blighted_string.split(',')
            for i in blighted_list:
                if i:
                    train_img_names.append(i)
                    train_img_label.append('B')
    """
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model, train_img_names

def renderLabel(form):
    """
    prepairs a render_template to show the label.html web page.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    queue = session['queue']
    img = queue.pop()
    session['queue'] = queue
    return render_template(url_for('label'), form = form, picture = img, confidence = session['confidence'])

def initializeAL(form, confidence_break = .7):
    """
    Initializes the active learning model and sets up the webpage with everything needed to run the application.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    confidence_break : number
        How confident the model is.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    preprocess = DataPreprocessing(True)
    ml_classifier = RandomForestClassifier()
    data = getData()
    al_model = Active_ML_Model(data, ml_classifier, preprocess)

    session['confidence'] = 0
    session['confidence_break'] = confidence_break
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['queue'] = list(al_model.sample.index.values)

    return renderLabel(form)


"""Look to make another initializeAL but in the case that a user is logged in"""



def getNextSetOfImages(form, sampling_method):
    """
    Uses a sampling method to get the next set of images needed to be labeled.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    sampling_method : SamplingMethods Function
        function that returns the queue and the new test set that does not contain the queue.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    data = getData()
    ml_model, train_img_names = createMLModel(data)
    test_set = data[data.index.isin(train_img_names) == False]

    session['sample_idx'], session['test'] = sampling_method(ml_model, test_set, 5)
    session['queue'] = session['sample_idx'].copy()

    return renderLabel(form)

def prepairResults(form):
    """
    Creates the new machine learning model and gets the confidence of the machine learning model.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the appropriate webpage based on new confidence score.
    """
    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))

    if session['train'] != None:
        session['train'] = session['train'] + session['sample']
    else:
        session['train'] = session['sample']

    data = getData()
    ml_model, train_img_names = createMLModel(data)

    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    if session['confidence'] < session['confidence_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic, blight_user = blight_pic, healthNum_user = len(health_pic), blightNum_user = len(blight_pic))
    else:
        test_set = data.loc[session['test'], :]
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        
        if current_user.is_authenticated:
            """If there is a user logged in we'll try and save their slections for healthy and blighted pictures to our database so that later the user can pick up where they left off
               Though since we are only storing the image names we'll need to regenerate the ML_Model when the want to continue
              The names will be stored as a string sepearted by ',' This will also be done at alater points to update their selected lists
              Since this needs to be done in strings we need two fields for each healthy and blighted, this will result in almost duplicated code
              One final note, we could store the form class directly but this will cause issues when we try to store the feedback selections since they don't use form"""
            
            """first find current user"""
            user = User.query.filter_by(username = current_user.username).first()
            
            """if a user already has a list of image names for data then we need to append to it instead of overwritting it
               One thing needed to do if we append list is we need to remove duplicates
               To do this we are turning it back into a list and subtracting the new list from that list and appending the result
               This should leave us with the new list without the duplciates then we reconvert back to string"""



            """if user has no confidence data we simply set names to our list from the form
            Originally this was an if/else statement but for some reason python wouldn't work that way
            Not sure why it wouldn't it would spit up errors on teh else: statement"""
            
            """we need to check if users pics exist"""
            if health_pic_user:
                health_pic_user_names = ",".join(health_pic_user)
            else:
                health_pic_user_names = ""
            if blight_pic_user:
                blighted_pic_user_names = ",".join(blight_pic_user)
            else:
                blighted_pic_user_names = ""


            """if user has confidence data we start converting and appending ignoring duplicates
               While I do not expect to run into any duplciates and also don't know if having them would muck up the machine learning I think it would be best to err on the side of caution"""
            if Confidence.query.filter_by(user_id = user.id).first():
                """get healthy data images and append ignoring duplicates"""
                original_healthy_string = Confidence.query.filter_by(user_id = user.id).first().healthy_data
                original_healthy_list = original_healthy_string.split(",")
                
                in_original_healthy = set(original_healthy_list)
                in_new_healthy = set(health_pic_user)
                in_new_not_original_healthy = in_new_healthy - in_original_healthy

                result_healthy = original_healthy_list + list(in_new_not_original_healthy)
                health_pic_user_names = ",".join(result_healthy)

                """get blighted data images and append ignoring duplicates"""
                original_blighted_string = Confidence.query.filter_by(user_id = user.id).first().blighted_data
                original_blighted_list = original_blighted_string.split(",")
                
                in_original_blighted = set(original_blighted_list)
                in_new_blighted = set(blight_pic_user)
                in_new_not_original_blighted = in_new_blighted - in_original_blighted

                result_blighted = original_blighted_list + list(in_new_not_original_blighted)
                blighted_pic_user_names = ",".join(result_blighted)

                """Clear original data in database"""
                db.session.delete(Confidence.query.filter_by(user_id = user.id).first())
                db.session.commit()

            """create new database data and commit"""
            user_data = Confidence(healthy_data = health_pic_user_names, blighted_data = blighted_pic_user_names, creator = user)
            db.session.add(user_data)
            db.session.commit()
            """now we render the page using the names in the database, since we've already pulled them we should just changed whats passed to the render_template
               This will need to be in list form so we change the string to list via string split"""
            if health_pic_user_names:
                health_pic_database = health_pic_user_names.split(",")
            else:
                health_pic_database = []

            if blighted_pic_user_names:
                blight_pic_database = blighted_pic_user_names.split(",")
            else:
                blight_pic_database = []

            return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_database, blight_user = blight_pic_database, healthNum_user = len(health_pic_database), blightNum_user = len(blight_pic_database), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_database)+len(blight_pic_database)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_database)+len(blight_pic_database)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)
        
        else: 
            return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)


@app.route("/", methods=['GET'])
@app.route("/index.html",methods=['GET'])
def home():
    """
    Operates the root (/) and index(index.html) web pages.
    """
    session.pop('model', None)
    return render_template('index.html')

@app.route("/leaderboards.html",methods=['GET', 'POST'])
def leaderboards():
    return render_template('leaderboards.html')

@app.route("/label.html",methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    """
    form = LabelForm()
    

    if current_user.is_authenticated:
        user = User.query.filter_by(username = current_user.username).first()
        if Confidence.query.filter_by(user_id = user.id).first():     
            """Here we need to initialize our ML however there's an issue with using the code made by the previous students as a lot of it is hard coded and doesn't
               expect to be fed images like this so we need to do it all by hand so we cant use initializeAL
               
               To do this look to ML_Class.py and how it uses test train and sample"""  
            """Here we are pulling the number of images stored in the database to be appened onto labels for the model to be created with"""
            healthy_string = Confidence.query.filter_by(user_id = user.id).first().healthy_data
            healthy_list = healthy_string.split(',')
            for i in healthy_list:
                if i:
                    session['labels'].append('H')

            blighted_string = Confidence.query.filter_by(user_id = user.id).first().blighted_data
            blighted_list = blighted_string.split(',')
            for i in blighted_list:
                if i:
                    session['labels'].append('B')

            """Here we pass a blank form since we don't gather any information from on and have already appended info into labels"""
            return prepairResults(form)
    
    if 'model' not in session:#Start
        return initializeAL(form, .7)
    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return render_template('label.html', form = form)

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    """
    Operates the login(login.html) web page.
    """
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('home'))
    return render_template('login.html', title='Sign In', form = form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registerd!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/intermediate.html",methods=['GET'])
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    return render_template('intermediate.html')

@app.route("/final.html",methods=['GET'])
def final():
    """
    Operates the final(final.html) web page.
    """
    return render_template('final.html')

@app.route("/feedback/<h_list>/<u_list>/<h_conf_list>/<u_conf_list>",methods=['GET'])
def feedback(h_list,u_list,h_conf_list,u_conf_list):
    """
    Operates the feedback(feedback.html) web page.
    """
    h_feedback_result = list(h_list.split(","))
    u_feedback_result = list(u_list.split(","))
    h_conf_result = list(h_conf_list.split(","))
    u_conf_result = list(u_conf_list.split(","))
    h_length = len(h_feedback_result)
    u_length = len(u_feedback_result)

    """Here we should store the selected images for storing in the database"""
    if current_user.is_authenticated:
        user = User.query.filter_by(username = current_user.username).first()
        if Confidence.query.filter_by(user_id = user.id).first():
            original_healthy_string = Confidence.query.filter_by(user_id = user.id).first().healthy_data

            healthy_pic_user_names = ""
            if original_healthy_string and (u_list != 'null'):
                original_healthy_list = original_healthy_string.split(",")
                in_original_healthy = set(original_healthy_list)
                in_new_healthy = set(u_feedback_result)
                in_new_not_original_healthy = in_new_healthy - in_original_healthy

                for i in (list(in_new_not_original_healthy)):
                    original_healthy_list.append(i)
                healthy_pic_user_names = ",".join(original_healthy_list)
            elif original_healthy_string:
                healthy_pic_user_names = original_healthy_string
            elif u_list:
                healthy_pic_user_names = u_list

            """get blighted data images and append ignoring duplicates"""
            original_blighted_string = Confidence.query.filter_by(user_id = user.id).first().blighted_data

            blighted_pic_user_names = ""
            if original_blighted_string and (h_list != 'null'):
                original_blighted_list = original_blighted_string.split(",")
                in_original_blighted = set(original_blighted_list)
                in_new_blighted = set(h_feedback_result)
                in_new_not_original_blighted = in_new_blighted - in_original_blighted

                for i in (list(in_new_not_original_blighted)):
                    original_blighted_list.append(i)
                blighted_pic_user_names = ",".join(original_blighted_list)
            elif original_blighted_string:
                blighted_pic_user_names = original_blighted_string
            elif h_list:
                blighted_pic_user_names = h_list

            """Clear original data in database"""
            db.session.delete(Confidence.query.filter_by(user_id = user.id).first())
            db.session.commit()

        """create new database data and commit"""

        user_data = Confidence(healthy_data = healthy_pic_user_names, blighted_data = blighted_pic_user_names, creator = user)
        db.session.add(user_data)
        db.session.commit()

    return render_template('feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)

#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)