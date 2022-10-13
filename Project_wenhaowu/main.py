#from cProfile import run
from kivy.core.window import Window
#from kivy.properties import StringProperty
from kivy.app import App
from kivy.uix.image import Image
from kivymd.uix.textfield import MDTextFieldRound
from kivy.uix.button import Button
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from datetime import datetime
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import Screen, NoTransition, CardTransition, ScreenManager
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
#from kivymd.uix.behaviors import RoundedRectangularElevationBehavior
from kivymd.uix.button import MDRectangleFlatIconButton
from kivymd.uix.textfield import MDTextFieldRect
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.input.providers.mouse import MouseMotionEvent
from kivymd.uix.menu import MDDropdownMenu
#from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
from kivymd.uix.button import MDRoundFlatButton
# from gardennotification import Notification
from plyer import notification, accelerometer
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
import pync 
#import time
import numpy as np
import random
from kivy.clock import Clock


#####################  SQL imports  ################################
import sqlite3
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# os.remove('trainer.db')
engine = create_engine('sqlite:///LoginDetails.db', echo=True)
Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()
#################################  SQL  ####################################

##### AI #####
import tensorflow as tf
#import keras

ACTIVITIES = {
    0: 'sitting',
    1: 'walking',
    2: 'running',
}

def load_model():
    model = tf.keras.models.load_model('model')
    return model

def predict(model, data):
    k = model.predict([data])
    predict_value = ACTIVITIES[np.argmax(k)]
    return predict_value
### AI ####
Sensor_values = []
m = load_model()

g_sensor_reachable = False


class User(Base):
    __tablename__ = "user"
    email = Column(String, primary_key=True)
    password = Column(String, nullable=False)
    name = Column(String)
    age = Column(String)
    weight = Column(String)
    uheight = Column(String)
    gender = Column(String)
    jp = Column(String)

    def __init__(self, email, password, name='', age='', weight='', uheight = '' ,gender='', jp=''):
        self.email = email
        self.password = password
        self.name = name
        self.age = age
        self.weight = weight
        self.uheight = uheight
        self.gender = gender
        self.jp = jp


class MovementAnalysis(Base):
    __tablename__ = "movement"

    email = Column(String, ForeignKey(User.email), primary_key=True, nullable=False)
    timestamp = Column(String, nullable=False, primary_key=True)
    mvmntprofile = Column(Integer)
    

    def __init__(self, email, timestamp, mvmntprofile):
        self.email = email
        self.timestamp = timestamp
        self.mvmntprofile = mvmntprofile

Base.metadata.create_all(engine) 

############################################################################


##############################################################################


logged_in_user = None



class LoginScreen(Screen):

    def login(self):
        email = ObjectProperty(None)
        password = ObjectProperty(None)
        email_input = self.email.text
        password_input = self.password.text
        print(email_input)
        print(password_input)
        valid = session.query(User).filter(
            User.email == email_input, User.password == password_input).count()
        if valid == 0 or (email_input == '' or password_input == ''):
            # if valid == 0:
            print("Invalid email or password")
        else:
            result = session.query(User).filter(
                User.email == email_input, User.password == password_input)
            for user in result:
                print(f"Welcome {user.name}!")
                global logged_in_user
                logged_in_user = user.email
                self.manager.current = "main"

        # self.manager.current="main"
        #result = session.query(User).filter(User.email == email_input).all()


class SignUpScreen(Screen):
    def registerBtn(self):
        email_input = self.email.text
        password_input = self.password.text
        password_confirmation_input = self.password_confirmation.text
        username_input = self.username.text
        age_input = self.age.text
        weight_input = self.weight.text
        uheight_input = self.uheight.text
        gender_input = self.gender.text
        jp_input = self.jp.text

        if password_confirmation_input != password_input:
            print("Password unmatch!!")
        else:
            data = User(email_input, password_input, username_input,
                        age_input, weight_input, uheight_input, gender_input, jp_input)
            session.add(data)
            session.commit()

class AcknowledgeScreen(Screen):
    pass

class MainScreen(Screen):
    # def switch_click(self, switchObject, switchValue):
    #     if (switchValue):
    #         self.ids.my_label.text = "You are now checked-in"
    #     else:
    #         self.ids.my_label.text = "You are now checked-out"
    show_notif = True
    motiv_task_list = []
    movement_profile = ObjectProperty(None)
    actual_movement = ''
    def notif_switch_click(self, switchObject, switchValue):
        # show_notif = True
        if(switchValue):
            self.ids.notification_switch_label.text = "Notification ON"
            self.show_notif = True
            self.ids.motiv_checkbox_1.disabled = False
            self.ids.motiv_checkbox_2.disabled = False
            self.ids.motiv_checkbox_3.disabled = False
            self.ids.motiv_checkbox_4.disabled = False
            self.ids.motiv_checkbox_5.disabled = False
        else:
            self.ids.notification_switch_label.text = "Notification OFF"
            self.show_notif = False
            self.motiv_task_list = []
            self.ids.motiv_checkbox_1.active = False
            self.ids.motiv_checkbox_2.active = False
            self.ids.motiv_checkbox_3.active = False
            self.ids.motiv_checkbox_4.active = False
            self.ids.motiv_checkbox_5.active = False
            self.ids.motiv_checkbox_1.disabled = True
            self.ids.motiv_checkbox_2.disabled = True
            self.ids.motiv_checkbox_3.disabled = True
            self.ids.motiv_checkbox_4.disabled = True
            self.ids.motiv_checkbox_5.disabled = True
    def get_info(self):
        
        global logged_in_user
        result = session.query(User).filter(User.email == logged_in_user)
        u_age = ''
        u_gender = ''
        for user in result:
            #self.email_label.text = user.email
            #self.username_label.text = user.name
            #self.age.text = user.age

            u_age = user.age
            #self.weight_label.text = user.weight
            #self.gender.text = user.gender
            u_gender = user.gender
            u_jp = user.jp
        # u_age = user.age
        # u_gender = user.gender
        u_age = int(u_age)
        u_jp = str(u_jp)
        
    #def check_body_info(self, u_age, job, u_gender, actual_movement_profile):
        # motivation_task = []
        # motiv_task1 = []
        # motiv_task2 = []
        # motiv_task3 = []
        # motiv_task4 = []
        # motiv_task5 = []
        if u_jp == "software engineer":
            if u_gender == "Male":
                if (u_age) > 18 and (u_age) < 40:
                    motivation_task = [
                        "software 18-40: male run1",
                        "software 18-40: male run2",
                        "software 18-40: male run3",
                        "software 18-40: male run4",
                        "software 18-40: male run5"
                    ]
                else:
                    motivation_task = [
                        "software 40+: male run1",
                        "software 40+: male run2",
                        "software 40+: male run3",
                        "software 40+: male run4",
                        "software 40+: male run5"
                    ]
            elif u_gender == "Female":
                if (u_age) > 18 and (u_age) < 40:
                    motivation_task = [
                        "software 18-40: female run1",
                        "software 18-40: female run2",
                        "software 18-40: female run3",
                        "software 18-40: female run4",
                        "software 18-40: female run5"
                    ]
                else:
                    motivation_task = [
                        "software 40+: female run1",
                        "software 40+: female run2",
                        "software 40+: female run3",
                        "software 40+: female run4",
                        "software 40+: female run5"
                    ]
        else:
            if u_gender == "Male":
                if (u_age) > 18 and (u_age) < 40:
                    motivation_task = [
                        "not software 18-40: male run1",
                        "not software 18-40: male run2",
                        "not software 18-40: male run3",
                        "not software 18-40: male run4",
                        "not software 18-40: male run5"
                    ]
                else:
                    motivation_task = [
                        "not software 40+: male run1",
                        "not software 40+: male run2",
                        "not software 40+: male run3",
                        "not software 40+: male run4",
                        "not software 40+: male run5"
                    ]
            elif u_gender == "Female":
                if (u_age) > 18 and (u_age) < 40:
                    motivation_task = [
                        "not software 18-40: female run1",
                        "not software 18-40: female run2",
                        "not software 18-40: female run3",
                        "not software 18-40: female run4",
                        "not software 18-40: female run5"
                    ]
                else:
                    motivation_task = [
                        "not software 40+: female run1",
                        "not software 40+: female run2",
                        "not software 40+: female run3",
                        "not software 40+: female run4",
                        "not software 40+: female run5"
                    ]


        self.motiv_task1.text = motivation_task[0]
        self.motiv_task2.text = motivation_task[1]
        self.motiv_task3.text = motivation_task[2]
        self.motiv_task4.text = motivation_task[3]
        self.motiv_task5.text = motivation_task[4]
        #print(motiv_task1, motiv_task2, motiv_task3, motiv_task4, motiv_task5)

    def close_dialog(self, obj):
        self.dialog.dismiss()

    def show_notification_alert_dialog(self):
        # if not self.dialog:
        #     self.dialog = MDDialog()
        
        self.dialog = MDDialog(
            title = "Error",
            text = "Your Operating System doesn't support notification, sorry.",
            buttons = [MDFlatButton(text = "OK", 
            # text_color = (1,1,1,1),
            on_release = self.close_dialog)]
        )

        self.dialog.open()

    def notification_text(self, value, instance):
        if self.show_notif == True:
            if value== True:
                self.motiv_task_list.append(instance)
            else:
                self.motiv_task_list.remove(instance)
        else:
            pass
        print(self.motiv_task_list)
    def pop_up_notification(self):
        str_motiv_task_list = "Please don't forget to do: "
        for notifications in self.motiv_task_list:
            str_motiv_task_list = str_motiv_task_list + notifications + " ,and "
        
        str_motiv_task_list = str_motiv_task_list[:-6]
        if self.show_notif == True:
            try:
                notification.notify(title='test', message=str_motiv_task_list, timeout=3)
                # time.sleep(10)
            except:
                pass
            try:
                pync.notify(str_motiv_task_list, title='Title')
                # time.sleep(10)
            except:
                self.show_notification_alert_dialog()
        elif self.show_notif == False:
            pass
            # self.ids.motiv_checkbox_1.active = False
            # self.ids.motiv_checkbox_2.active = False
            # self.ids.motiv_checkbox_3.active = False
            # self.ids.motiv_checkbox_4.active = False
            # self.ids.motiv_checkbox_5.active = False
            

    def check_in_deactivate(self, value):
        if value == False:
            self.ids.check_in_button.disabled = True
            self.ids.check_out_button.disabled = False
            self.start_movement_analysis()

    def check_out_deactivate(self, value):
        if value == False:
            self.ids.check_in_button.disabled = False
            self.ids.check_out_button.disabled = True
            self.event.cancel()

    def callback_get_movement_data(self,dt):
    
    #global g_time_in_movement
    

        if g_sensor_reachable:
            Sensor_values.append([accelerometer.accelerometer[0],accelerometer.accelerometer[1],accelerometer.accelerometer[2]])
        else:
            Sensor_values.append([random.random(),random.random(),random.random()])
        print(self.movement_profile)
        if self.movement_profile not in ACTIVITIES.keys():
            self.movement_profile = "calculating"

        if (len(Sensor_values) == 5):
        
            print(predict(m,Sensor_values))
            print(Sensor_values)
            last_movement_profile = self.movement_profile
            mp = predict(m,Sensor_values)
            self.movement_profile = mp
            # write_movement_data(get_user_data_with_email(self.email)["user_id"],datetime.now().replace(microsecond=0),ACTIVITIES[mp])
            buffer  = MovementAnalysis(logged_in_user, datetime.now(), list(ACTIVITIES.values()).index(mp))
            print(list(ACTIVITIES.values()).index(mp))
            self.actual_movement.text = "You are now: " + str(self.movement_profile)
            session.add(buffer)
            session.commit()
            Sensor_values.clear()
            
            if last_movement_profile == self.movement_profile:
                g_time_in_movement += 1
            else:
                g_time_in_movement = 0
        
            if g_time_in_movement == 3:
                if self.notification:
                    notification.notify(title='test', message='You have stayed in the same movement for 2 hours')
                    g_time_in_movement = 0
        
    def start_movement_analysis(self):

        self.event = Clock.schedule_interval(self.callback_get_movement_data, 1/5.)      
        # age = date.today().year - datetime.strptime(g_user_birthdate, '%d.%m.%Y').year

        if self.movement_profile not in ['sitting','walking','running']:
            sql_movement_profile = 0
        else:
            sql_movement_profile = ACTIVITIES[self.movement_profile]

class SettingsScreen(Screen):
    pass


class GraphScreen(Screen):
    pass


class NotificationScreen(Screen):
    pass


class UserProfileScreen(Screen):

    # we need to get the user value from the database
    # current_user= session.query(User).filter()

    def on_pre_enter(self):
        self.callback()

    def callback(self):
        global logged_in_user
        result = session.query(User).filter(User.email == logged_in_user)
        for user in result:
            self.email_label.text = user.email
            self.username_label.text = user.name
            self.age_label.text = user.age
            self.weight_label.text = user.weight
            self.uheight_label.text = user.uheight
            self.gender_label.text = user.gender
            self.jp_label.text = user.jp


class EditUserProfileScreen(Screen):

    def on_pre_enter(self):
        self.callback()

    def callback(self):
        global logged_in_user
        result = session.query(User).filter(User.email == logged_in_user)
        for user in result:
            self.chg_username.text = user.name
            self.chg_age.text = user.age
            self.chg_weight.text = user.weight
            self.chg_uheight.text = user.uheight
            self.chg_gender.text = user.gender
            self.chg_jp.text = user.jp

    def savechanges(self):
        username_input = self.chg_username.text
        age_input = self.chg_age.text
        weight_input = self.chg_weight.text
        uheight_input = self.chg_uheight.text
        gender_input = self.chg_gender.text
        jp_input = self.chg_jp.text
        session.query(User).filter(User.email == logged_in_user).update(
            {'name': username_input, 'age': age_input, 'weight': weight_input, 'uheight': uheight_input, 'gender': gender_input, 'jp': jp_input})
        session.commit()
        self.manager.current = "settings"


sm = ScreenManager()
# sm.add_widget(MDScreen(name="md"))
sm.add_widget(LoginScreen(name="login"))
sm.add_widget(SignUpScreen(name="signup"))
sm.add_widget(MainScreen(name="main"))
sm.add_widget(SettingsScreen(name="settings"))
sm.add_widget(GraphScreen(name="graph"))
sm.add_widget(NotificationScreen(name="notify"))
sm.add_widget(UserProfileScreen(name="userprofile"))
sm.add_widget(EditUserProfileScreen(name="edituserprofile"))
# class ScreenManager(ScreenManager):
#   pass


# class ProfileCard(FloatLayout, FakeRectangleElevationBehaviour):
#   pass

# class MovementAnalysis(ProfileCard):
#   pass


Window.size = (320, 550)

# GUI= Builder.load_file("main.kv") #Make Sure this is after all clas definitions!


import pkg_resources
import prettytable

def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'

def print_packages_and_licenses():
    t = prettytable.PrettyTable(['Package', 'License'])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        t.add_row((str(pkg), get_pkg_license(pkg)))
    print(t)


if __name__ == "__main__":
    print_packages_and_licenses()

class MainApp(MDApp):

    def build(self):
        #builder = builder.load_file("main.kv")
        # return #builder
        pass


if __name__ == '__main__':
    MainApp().run()
