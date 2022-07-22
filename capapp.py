import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

df=pd.read_csv("aug_train.csv")
df.drop('enrollee_id', axis=1, inplace=True)
df.drop('city', axis=1, inplace=True)

def gender_to_numeric(x):
    if x=='Female': return 2
    if x=='Male':   return 1
    if x=='Other':   return 0
    
def rel_experience(x):
    if x=='Has relevent experience': return 1
    if x=='No relevent experience':   return 0
    
def enrollment(x):
    if x=='no_enrollment'   : return 0
    if x=='Full time course':   return 1 
    if x=='Part time course':   return 2 
    
def edu_level(x):
    if x=='Graduate'       :   return 0
    if x=='Masters'        :   return 1 
    if x=='High School'    :   return 2 
    if x=='Phd'            :   return 3 
    if x=='Primary School' :   return 4 
    
def major(x):
    if x=='STEM'                   :   return 0
    if x=='Business Degree'        :   return 1 
    if x=='Arts'                   :   return 2 
    if x=='Humanities'             :   return 3 
    if x=='No Major'               :   return 4 
    if x=='Other'                  :   return 5 
    
def experience(x):
    if x=='<1'      :   return 0
    if x=='1'       :   return 1 
    if x=='2'       :   return 2 
    if x=='3'       :   return 3 
    if x=='4'       :   return 4 
    if x=='5'       :   return 5
    if x=='6'       :   return 6
    if x=='7'       :   return 7
    if x=='8'       :   return 8 
    if x=='9'       :   return 9 
    if x=='10'      :   return 10 
    if x=='11'      :   return 11
    if x=='12'      :   return 12
    if x=='13'      :   return 13 
    if x=='14'      :   return 14 
    if x=='15'      :   return 15 
    if x=='16'      :   return 16
    if x=='17'      :   return 17
    if x=='18'      :   return 18
    if x=='19'      :   return 19 
    if x=='20'      :   return 20 
    if x=='>20'     :   return 21 
    
def company_t(x):
    if x=='Pvt Ltd'               :   return 0
    if x=='Funded Startup'        :   return 1 
    if x=='Early Stage Startup'   :   return 2 
    if x=='Other'                 :   return 3 
    if x=='Public Sector'         :   return 4 
    if x=='NGO'                   :   return 5 
    
def company_s(x):
    if x=='<10'          :   return 0
    if x=='10/49'        :   return 1 
    if x=='100-500'      :   return 2 
    if x=='1000-4999'    :   return 3 
    if x=='10000+'       :   return 4 
    if x=='50-99'        :   return 5 
    if x=='500-999'      :   return 6 
    if x=='5000-9999'    :   return 7
    
def last_job(x):
    if x=='never'        :   return 0
    if x=='1'            :   return 1 
    if x=='2'            :   return 2 
    if x=='3'            :   return 3 
    if x=='4'            :   return 4 
    if x=='>4'           :   return 5
    
df['gender'] = df['gender'].apply(gender_to_numeric)
df['relevent_experience'] = df['relevent_experience'].apply(rel_experience)
df['enrolled_university'] = df['enrolled_university'].apply(enrollment)
df['education_level'] = df['education_level'].apply(edu_level)
df['major_discipline'] = df['major_discipline'].apply(major)
df['experience'] = df['experience'].apply(experience)
df['company_type'] = df['company_type'].apply(company_t)
df['company_size'] = df['company_size'].apply(company_s)
df['last_new_job'] = df['last_new_job'].apply(last_job)

df['gender'] = df['gender'].fillna((df['gender'].mean()))
df['enrolled_university'] = df['enrolled_university'].fillna((df['enrolled_university'].mean()))
df['major_discipline'] = df['major_discipline'].fillna((df['major_discipline'].mean()))
df['company_size'] = df['company_size'].fillna((df['company_size'].mean()))
df['company_type'] = df['company_type'].fillna((df['company_type'].mean()))
df['company_type'] = df['company_type'].fillna((df['company_type'].mean()))
df['education_level'] = df['education_level'].fillna((df['education_level'].mean()))
df['last_new_job'] = df['last_new_job'].fillna((df['last_new_job'].mean()))
df['experience'] = df['experience'].fillna((df['experience'].mean()))

df = df[df['target'].notna()]

X = df.drop(['target'], axis = 1)
y = df['target']

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.33)

sc=StandardScaler()
sc.fit(np.array(X_train))


pickle_in = open("finalized_model.sav", "rb")
model_svc = pickle.load(pickle_in)

def predictres(l):
    l = np.array(l)
    l=l.reshape(1, -1)
    l = sc.transform(l)
    model_prediction = model_svc.predict(l)
    return model_prediction


st.title("L&D Analytics")

gender = st.radio("Gender",('Male', 'female'))
if gender =='Male':
    gender=1
else:
    gender=2

rel_exp = st.radio("Has relevent experience?", ('Yes', 'No'))
if rel_exp =='Yes':
    rel_exp = 1
else:
    rel_exp=0

cdi = st.number_input(
"Enter city development index(CDI)",
min_value=0.0,
max_value=1.0,
step=1e-6,
format="%.3f")

enroll = st.radio("Type of university enrollment", ('None', 'Part time', 'Full time'))
if enroll=='None':
    enroll=0
elif enroll=='Part time':
    enroll =2
else:
    enroll=1

edulvl = st.radio("Education level: ", ('Graduate', 'Masters', 'High School', 'Phd', 'Primary School'))
if edulvl=='Graduate':
    edulvl=0
elif edulvl=='Masters':
    edulvl=1
elif edulvl=='High School':
    edulvl=2
elif edulvl=='Phd':
    edulvl=3
else:
    edulvl=4

major = st.radio("Education level: ", ('STEM', 'Business', 'Arts', 'Humanities', 'None','Other'))
if major=='STEM':
    major=0
elif major=='Business':
    major=1
elif major=='Arts':
    major=2
elif major=='Humanities':
    major=3
elif major=='None':
    major=4
else:
    major=5

exp = st.number_input('Enter your years of experience', min_value=0)
if exp <1:
    exp = 0
elif exp == 1:
    exp= 1
elif exp == 2:
    exp=2
elif exp == 3:
    exp= 3
elif exp == 4:
    exp= 4
elif exp == 5:
    exp= 5
elif exp == 6:
    exp= 6
elif exp == 7:
    exp= 7
elif exp == 8:
    exp= 8
elif exp == 9:
    exp= 9
elif exp == 10:
    exp= 10
elif exp == 11:
    exp= 11
elif exp == 12:
    exp= 12
elif exp == 13:
    exp= 13
elif exp == 14:
    exp= 14
elif exp == 15:
    exp= 15
elif exp == 16:
    exp= 16
elif exp == 17:
    exp= 17
elif exp == 18:
    exp= 18
elif exp == 19:
    exp= 19
elif exp == 20:
    exp= 20
else:
    exp=21

ct = st.radio("Current company type: ", ('pvt ltd', 'funded startup', 'early startup', 'other', 'public','ngo'))
if ct =='pvt ltd':
    ct= 0
elif ct == 'funded startup':
    ct= 1
elif ct == 'early startup':
    ct= 2
elif ct == 'other':
    ct= 3
elif ct == 'public':
    ct= 4
else:
    ct= 5

cs = st.radio("Current company size: ", ('<10', '10-49', '100-500', '1000-4999', '10000+','50-99','500-999','5000-9999'))
if cs =='<10':
    cs= 0
elif cs == '10-49':
    cs= 1
elif cs == '100-500':
    cs= 2
elif cs == '1000-4999':
    cs= 3
elif cs == '10000+':
    cs= 4
elif cs == '50-99':
    cs= 5
elif cs == '500-999':
    cs= 6
else:
    cs= 7

lnj = st.radio("Difference in years between previous job and current job: ", ('Never', '1', '2', '3', '4','>4'))
if lnj =='Never':
    lnj= 0
elif lnj == '1':
    lnj= 1
elif lnj == '2':
    lnj= 2
elif lnj == '3':
    lnj= 3
elif lnj == '4':
    lnj= 4
else:
    lnj= 5

th = st.number_input('Enter the number of hours trained]', min_value=0)

xa = pd.Series([cdi, gender, rel_exp, enroll, edulvl, major, exp, cs, ct, lnj, th])


if st.button("Predict"):
    predictres(xa)
    if prediction == 0:
        st.error('They are likely to leave the job')
    elif prediction == 1:
        st.success('They are likely to take the job')

        
if __name__ == '__main__':
    main()        



