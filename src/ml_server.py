import os
import pickle
import datetime

import numpy as np
from ensembles import RandomForestMSE, GradientBoostingMSE

# import plotly
# import plotly.subplots
# import plotly.graph_objects as go
# from shapely.geometry.polygon import Point
# from shapely.geometry.polygon import Polygon

import pandas as pd

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, flash
from flask import render_template, redirect
from wtforms import validators

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField, SelectField, TextAreaField, IntegerField, DecimalField

# from utils import polygon_random_point


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'mmp'
data_path = './../data'
Bootstrap(app)
messages = []

class Message:
    header = ''
    text = ''

class RF(FlaskForm):
    n_estimators = IntegerField('n_estimators:', validators=[DataRequired(), validators.NumberRange(min=1, max=5000)])
    depth = IntegerField('max_depth:', [validators.NumberRange(min=0, max=50)])
    fss = IntegerField('feature_subsample_size:', [DataRequired(), validators.NumberRange(min=1, max=30000)])
    data_train = FileField('Train data', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    data_val = FileField('Validation data', validators=[
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField("Send")

class GB(FlaskForm):
    n_estimators = IntegerField('n_estimators:', [DataRequired(),validators.NumberRange(min=1, max=5000)])
    depth = IntegerField('max_depth:', [validators.NumberRange(min=0, max=50)])
    fss = IntegerField('feature_subsample_size:', [DataRequired(), validators.NumberRange(min=1, max=30000)])
    lr = DecimalField('learning_rate:', [DataRequired(),validators.NumberRange(min=0.001, max=1)])
    data_train = FileField('Train data', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    data_val = FileField('Validation data', validators=[
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField("Send")

class FileForm(FlaskForm):
    data_test = FileField('Test data', validators=[
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Predict')

@app.route('/')
@app.route('/index')
def index():
    return render_template('main_page.html')

model_list = []

@app.route('/RandomForest', methods=['GET', 'POST'])
def RF_model():
    global flag
    flag = False
    global X_train
    global y_train
    global X_val
    global y_val
    global val
    val = False
    model_RF = RF()
    model_list.append(model_RF)
    if model_RF.validate_on_submit():
        data_train1 = pd.read_csv(model_RF.data_train.data)
        if 'TARGET' not in data_train1:
            flash("В обучающем датасете нет столбца с именем 'TARGET'!")
            return redirect(url_for('RF_model'))
        if model_RF.fss.data > data_train1.shape[1]:
            flash("Размерность пространства признаков меньше, чем Вы выбрали!")
            return redirect(url_for('RF_model'))
        if model_RF.data_val.data.filename != "":
            data_val = pd.read_csv(model_RF.data_val.data)
            if 'TARGET' not in data_val:
                flash("В валидационном датасете нет столбца с именем 'TARGET'!")
                return redirect(url_for('RF_model'))
            y_val = np.array(data_val["TARGET"])
            X_val = np.array(data_val.drop("TARGET", axis=1))
            val = True
        else:
            y_val = None
            X_val = None
        y_train = np.array(data_train1["TARGET"])
        X_train = np.array(data_train1.drop("TARGET", axis=1))
        return redirect(url_for('get_predict', model = model_RF))
    return render_template('RF.html', title = 'Random Forest', form=model_RF)

@app.route('/GradientBoosting', methods=['GET', 'POST'])
def GB_model():
    global flag
    flag = False
    global X_train
    global y_train
    global X_val
    global y_val
    global val
    val = False
    model_GB = GB()
    model_list.append(model_GB)
    if model_GB.validate_on_submit():
        data_train1 = pd.read_csv(model_GB.data_train.data)
        if model_GB.fss.data > data_train1.shape[1] - 1:
            flash("Размерность пространства признаков меньше, чем Вы выбрали!")
            return redirect(url_for('GB_model'))
        if 'TARGET' not in data_train1:
            flash("В обучающем датасете нет столбца с именем 'TARGET'!")
            return redirect(url_for('GB_model'))
        if model_GB.data_val.data.filename != "":
            data_val = pd.read_csv(model_GB.data_val.data)
            if 'TARGET' not in data_val:
                flash("В валидационном датасете нет столбца с именем 'TARGET'!")
                return redirect(url_for('GB_model'))
            y_val = np.array(data_val["TARGET"])
            X_val = np.array(data_val.drop("TARGET", axis=1))
            val = True
        else:
            y_val = None
            X_val = None
        y_train = np.array(data_train1["TARGET"])
        X_train = np.array(data_train1.drop("TARGET", axis=1))
        return redirect(url_for('get_predict'))
    return render_template('GB.html', title = 'Gradient Boosting', form=model_GB)

@app.route('/choice_page', methods=['GET', 'POST'])
def get_predict():
    global flag
    global test
    global X_test
    test = FileForm()
    if test.validate_on_submit():
        X_test = np.array(pd.read_csv(test.data_test.data))
        return redirect(url_for('predict'))
    if not flag:
        global type
        global val
        global rmse_train
        global rmse_val
        global time
        global regressor
        type = False
        val = False
        if model_list[-1].depth.data == 0:
            model_list[-1].depth.data = None
        if isinstance(model_list[-1], RF):
            regressor = RandomForestMSE(n_estimators=model_list[-1].n_estimators.data,
                                        max_depth=model_list[-1].depth.data,
                                        feature_subsample_size=model_list[-1].fss.data)
        else:
            regressor = GradientBoostingMSE(n_estimators=model_list[-1].n_estimators.data,
                                            max_depth=model_list[-1].depth.data,
                                            feature_subsample_size=model_list[-1].fss.data,
                                            learning_rate=float(model_list[-1].lr.data))
            type = True
        rmse_val = 0
        regressor.fit(X_train, y_train, X_val, y_val)
        if X_val is not None:
            val = True
            rmse_val = regressor.rmse_val
        rmse_train = regressor.rmse_train
        time = regressor.times[-1]
        flag = True
    return render_template('choice_page.html', title = 'Choice', form=test)

@app.route('/info', methods=['GET', 'POST'])
def info():
    return render_template('info.html', title='Info',
                           form=model_list[-1],
                           type=type,
                           rmse_train=rmse_train,
                           rmse_val=rmse_val,
                           val=val,
                           time=time,
                           len=len(regressor.rmse_train))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global regressor
    global test
    global X_test
    y_pred = regressor.predict(X_test)
    return render_template('predict.html', title='Predict', y_pred=y_pred, len=len(y_pred))