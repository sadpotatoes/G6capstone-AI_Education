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

    """In order to stop an issue where the system asks the user to re label one of their selected images we need to make sure the queue is empty and skip calling RenderLabel"""

    if current_user.is_authenticated:
        user = User.query.filter_by(username = current_user.username).first()
        if Confidence.query.filter_by(user_id = user.id).first():
            session['queue'] = []
            
            img_labels = Confidence.query.filter_by(user_id = user.id).first().img_labels
            img_labels_list = img_labels.split(',')
            for i in img_labels_list:
                if i:
                    session['labels'].append(i)


            return prepairResults(form)


    return renderLabel(form)




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
    """Here if use is logged in and has a database entry then we need to append those to the session labels else append from latest form choice"""
    if form.choice.data:
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
            """For storing user selected labels in the database we need top first pull the current images in the sessions train data and the labels then add those to the database"""
            """first find current user"""
            user = User.query.filter_by(username = current_user.username).first()
            
            img_names = ""
            labels = ""
            
            
            """Clear original data in database"""
            if Confidence.query.filter_by(user_id = user.id).first():

                db.session.delete(Confidence.query.filter_by(user_id = user.id).first())
                db.session.commit()

            """create new database data and commit"""
            temp_img_names, temp_labels =  list(zip(*session['train']))
            
            img_names = ",".join(temp_img_names)
            labels = ",".join(temp_labels)
            user_data = Confidence(img_names = img_names, img_labels = labels, creator = user)
            db.session.add(user_data)
            db.session.commit()

            return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)
        
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

@app.route("/label.html",methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    """
    form = LabelForm()
    
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
            img_names = Confidence.query.filter_by(user_id = user.id).first().img_names
            labels = Confidence.query.filter_by(user_id = user.id).first().img_labels
            img_names_list = img_names.split(",")
            if u_list != 'null':
                for i in u_feedback_result:
                    img_names_list.append(i)
                    labels = labels + ',' + 'H'
            
            if h_list != 'null':
                for i in h_feedback_result:
                    img_names_list.append(i)
                    labels = labels + ',' + 'B'

            """Clear original data in database"""
            img_names = ",".join(img_names_list)
            db.session.delete(Confidence.query.filter_by(user_id = user.id).first())
            db.session.commit()

        """create new database data and commit"""

        user_data = Confidence(img_names = img_names, img_labels = labels, creator = user)
        db.session.add(user_data)
        db.session.commit()

    return render_template('feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)

#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)