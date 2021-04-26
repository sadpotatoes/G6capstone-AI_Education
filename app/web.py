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
from app.models import User, Confidence, ImageStats
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

    """In order to stop an issue where the system asks the user to re label one of their selected images we need to make sure the queue is empty and skip calling renderLabel"""

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
    """This is temp fix to find the largest confidence rate and use that model
    instead should findout why the ml model is generating three different models 
    and correct that <- After reviewing Scilearn and their methods for training it turns out they are stochastic in nature meaning
    they will always have a form of 'randomness' that can't really be 'fixed' since its part of the library

    One thing tht could be looked into instead of regenerating models is saving the models directly using something like pickel or joblib
    but these have their own issues. 

    ideally what should happen is we have a filesystem that we store the models in and have a database entry for each user pointing to their saved data in the system

    """
    confidence_test = 0
    for i in range(5):
        ml_model_temp, train_img_names_temp = createMLModel(data)
        conf_temp = np.mean(ml_model_temp.K_fold())
        if conf_temp > confidence_test:
            ml_model = ml_model_temp
            train_img_names = train_img_names_temp
            session['confidence'] = conf_temp
            confidence_test = conf_temp

    """
    ml_model, train_img_names = createMLModel(data)
    session['confidence'] = np.mean(ml_model.K_fold())
    """
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
            temp_img_names, temp_labels =  list(zip(*session['train']))
            """For some reason unzipping the session train was reversing the image names so we need to fix that"""
            list(temp_img_names).reverse()
            accuracy = 0.0
            img_names = ",".join(temp_img_names)
            labels = ",".join(temp_labels)
            """If user has information already stored then update it"""
            if Confidence.query.filter_by(user_id = user.id).first():
                c = Confidence.query.filter_by(user_id = user.id).first()
                c.img_labels = labels
                c.img_names = img_names

                """if they dont have information then create new table data"""
            else:
                user_data = Confidence(img_names = img_names, img_labels = labels, creator = user, accuracy_rate = accuracy, previous = '')
                db.session.add(user_data)
            """Commit changes"""
            db.session.commit()
            return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)
        
        else: 
            return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)

def findCorrect(max_num_images):
    """
    Finds the images the user labeled correctly and incorrectly

    Parameters
    ----------
    max_num_images integar for use in stoping the function early in case of finding previous accuracy rates

    Returns
    -------
    Lists accuracy, correct, incorrect, cor_label, inc_label
        and number of images
    """

    """These will hold image names that are correct and incorrect"""
    correct = []
    incorrect = []
    """These will hold image labels that are correct and incorrect"""
    cor_label = []
    inc_label = []
    """position in list"""
    position = 0
    if current_user.is_authenticated:
        user, c = pullUserData()
        if c:
            """If user hase stored data then pull from there"""
            temp_img_names, temp_labels = c.img_names.split(","), c.img_labels.split(",")
    else:
        """Since gamemode will be accessed from final.html we can make use of session[sample] """
        temp_img_names, temp_labels =  list(zip(*session['train']))
    """Need to pull data from site, since getData() removes the ground truth in its return value we can't use that code
    and we can't just add it to the return since that'll break some of the Machine learning algorithms"""
    s3 = boto3.client('s3')
    path = 's3://cornimagesbucket/csvOut.csv'

    data = pd.read_csv(path, index_col = 0, header = None)
    data.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    data_mod = data.astype({'8': 'int32','9': 'int32','10': 'int32','12': 'int32','14': 'int32'})
    """select just index and last column since the rest of the imformation doesn't matter for this task"""
    data = data_mod.iloc[:, -1:]
    """now to do the comparison we need to use a for loop for the list of images and use loc[image_name] to pull the row corresponding to that image
    then we compare the last column with the position in labels then append respectively""" 
    for i in temp_img_names:
        temp = data.loc[i, '16']
        if temp_labels[position] == temp:
            correct.append(i)
            cor_label.append(temp_labels[position])
        else:
            incorrect.append(i)
            inc_label.append(temp_labels[position])
        position += 1
        if position == max_num_images:
            break
    """find accuracy rate"""
    accuracy = (len(correct)/position)
    return accuracy, correct, incorrect, cor_label, inc_label, len(temp_img_names)

def pullUserData():
    """
    Returns the current user and their confidence data if applicable

    Parameters
    ----------
    None
    
    Returns
    -------
    SQLAlchemy datatypes for user and confidence tabels
    """
    """This function will be used to replace all the if/else statements throughout the code to clean it up a bit"""
    user = User.query.filter_by(username = current_user.username).first()
    if Confidence.query.filter_by(user_id = user.id).first():
        c = Confidence.query.filter_by(user_id = user.id).first()
    else:
        c = None

    return(user, c)

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
    """
    Operates the leaderboards(leaderboards.html) web page.
    """
    from sqlalchemy import desc
    """This pulls all data in the Confidence table from the database and sorts in descending order based on accuracy rate
    for the purposes of our leaderboard we only want to display top 20 so we need a breaking variable"""
    users = Confidence.query.order_by(desc(Confidence.accuracy_rate)).all()
    usernames = []
    accuracies = []
    num_images = []
    highest_correct_list = []
    num_times_cor_id = []
    highest_incorrect_list = []
    num_times_mis_id = []

    length_of_board = 0
    for i in users:
        usernames.append(User.query.filter_by(id = i.user_id).first().username)
        accuracies.append("{:.2%}".format(round(i.accuracy_rate,4)))
        temp = i.previous.split(",")
        num_images.append(temp[(len(temp) - 1)])
        length_of_board += 1
        """if we have 20 users selected for leader board then break"""
        if length_of_board >= 20:
            break

    top_correct_images = ImageStats.query.order_by(desc(ImageStats.cor_id_times)).limit(10).all()
    for i in top_correct_images:
        if i.cor_id_times != 0:
            highest_correct_list.append(i.img_name)
            num_times_cor_id.append(i.cor_id_times)

    top_incorrect_images = ImageStats.query.order_by(desc(ImageStats.mis_id_times)).limit(10).all()
    for i in top_incorrect_images:
        if i.mis_id_times != 0:
            highest_incorrect_list.append(i.img_name)
            num_times_mis_id.append(i.mis_id_times)
    
    return render_template('leaderboards.html', names = usernames, acc = accuracies, num_imgs = num_images, length = len(usernames), high = highest_correct_list, high_len = len(highest_correct_list), cor_id_times = num_times_cor_id, low = highest_incorrect_list, low_len = len(highest_incorrect_list), mis_id_times = num_times_mis_id)

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
    error = None
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid username or password")
            return redirect(url_for('login.html'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('home'))
    return render_template('login.html', title='Sign In', form = form, error_msg = error)

@app.route('/logout')
def logout():
    """
    Operates the logout button.
    """
    logout_user()
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Operates the register(register.html) web page.
    """
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

@app.route('/profile.html', methods=['GET', 'POST'])
def profile():
    """
    Operates the profile(profile.html) web page.
    """
    user = User.query.filter_by(username = current_user.username).first()
    if Confidence.query.filter_by(user_id = user.id).first():
        c = Confidence.query.filter_by(user_id = user.id).first() 
        accuracy, correct, incorrect, cor_label, inc_label, length = findCorrect(max_num_images=-1)
        
        len_imgs = len(correct) + len(incorrect)
        previous_image_selection = c.previous.split(",")
        len_previous = len(previous_image_selection)
        c.accuracy_rate = accuracy
        if c.previous == '':
            c.previous = str(length)
        else:
            temp_list = c.previous.split(",")
            if temp_list[-1] != str(length):
                c.previous = c.previous + ',' + str(length)
        db.session.commit()
        correct_length = len(correct)
        incorrect_length = len(incorrect)

    else:
        accuracy = 0 
        correct = None
        incorrect = None
        cor_label = None
        inc_label = None
        previous_image_selection = None
        correct_length = 0
        incorrect_length = 0
        len_imgs = 0
        len_previous = 0

    return render_template('profile.html', len_images = len_imgs, correct_list = correct, incorrect_list = incorrect, cor_label_list = cor_label, inc_label_list = inc_label, cor_length = correct_length, inc_length = incorrect_length, acc = "{:.2%}".format(round(accuracy,4)), prev_imgs = previous_image_selection, len_prev = len_previous-1)

@app.route('/previous/', methods=['GET', 'POST'])
def previousSelections():
    """
    Operates the previousSelections(previous.html) web page.
    """
    max_num_images = request.args.get('max', default='0', type=str)
    
    accuracy, correct, incorrect, cor_label, inc_label, length = findCorrect(max_num_images=int(max_num_images))
    return render_template('previous.html', correct_list = correct, incorrect_list = incorrect, cor_label_list = cor_label, inc_label_list = inc_label, cor_length = len(correct), inc_length = len(incorrect), acc = "{:.2%}".format(round(accuracy,4)))

@app.route("/clearData.html")
def clearData():
    """
    Operates the clear database button in the profile page.
    """
    user = User.query.filter_by(username = current_user.username).first()
    c = Confidence.query.filter_by(user_id = user.id).first()
    db.session.delete(c)
    db.session.commit()

    return redirect(url_for('profile'))

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
    """should look to move code out of page handler since it could be causing slow down"""
    if current_user.is_authenticated:
        user = User.query.filter_by(username = current_user.username).first()
        if Confidence.query.filter_by(user_id = user.id).first():
            c = Confidence.query.filter_by(user_id = user.id).first()

            img_names = c.img_names
            labels = c.img_labels
            img_names_list = img_names.split(",")
            if u_list != 'null':
                for i in u_feedback_result:
                    img_names_list.append(i)
                    labels = labels + ',' + 'H'
            
            if h_list != 'null':
                for i in h_feedback_result:
                    img_names_list.append(i)
                    labels = labels + ',' + 'B'

            """update data in database"""
            img_names = ",".join(img_names_list)
            c.img_names = img_names
            c.img_labels = labels
            db.session.commit()

    return render_template('feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)


@app.route("/gamemode",methods=['GET'])
def gamemode():
    """
    Operates the gamemode(gamemode.html) web page.
    """
    """find relavant information"""
    accuracy, correct, incorrect, cor_label, inc_label, length = findCorrect(max_num_images=-1)
    
    """if user is logged in then store their accuracy rate"""
    if current_user.is_authenticated:
        user = User.query.filter_by(username = current_user.username).first()
        if Confidence.query.filter_by(user_id = user.id).first():
            c = Confidence.query.filter_by(user_id = user.id).first()
            c.accuracy_rate = accuracy
            if c.previous == '':
                c.previous = str(length)
            else:
                temp_list = c.previous.split(",")
                if temp_list[-1] != str(length):
                    c.previous = c.previous + ',' + str(length)
            db.session.commit()
    for i in correct:
        image_info = ImageStats.query.get(i)
        if image_info:
            image_info.cor_id_times = image_info.cor_id_times + 1   
        else:
            image_info = ImageStats(img_name = i, cor_id_times = 1, mis_id_times = 0)
            db.session.add(image_info)
            
    for i in incorrect:
        image_info = ImageStats.query.get(i)
        if image_info:
            image_info.mis_id_times = image_info.mis_id_times + 1
        else:
            image_info = ImageStats(img_name = i, cor_id_times = 0, mis_id_times = 1)
            db.session.add(image_info)

    db.session.commit()

    return render_template('gamemode.html', correct_list = correct, incorrect_list = incorrect, cor_label_list = cor_label, inc_label_list = inc_label, cor_length = len(correct), inc_length = len(incorrect), acc = "{:.2%}".format(round(accuracy,4)))


#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)