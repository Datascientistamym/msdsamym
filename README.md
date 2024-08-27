1. Clearly communicate the problem and objective in business terms and how your solution will be used.
The objective of this project is to predict final grades of students, using this information we can identify which students require additional help to pass their subject. The business problem will be predicted by a list of each students attributes including family, age, and activites outside of school. A complete list will be given further into the project.
2. How should you frame this problem (supervised/unsupervised, online/offline, etc)? Briefly explain these terms since part of your audience is non-technical.
This is a supervised learning task because the model will be learning from labeled data. If the data wasn't labeled it would be considered unsupervised learning. Because the data set is static (does not change over time), it is considered offline and will not be performing real-time updates. If a project was online, it would be constantly making predictions such as stock market predictions.
3. Discuss the specific machine learning task that you are working on (regression/classification) and how it could solve the business problem. Briefly explain the difference since part of your audience is non-technical.
Making the choice depends on the data being used. Taking a glance at the CSV file provided, it shows that the data for our output target is discrete, meaning it is represented by integers, or, a number without decimal points. Because our target value is discrete, a Classification model will be used. If the value of the target attrubute had decimal placed (aka continuous data), a regression model would be used.
4. Identify the metrics that you will use to measure the model’s performance.
The metrics used will be R2 Score, Precision, Recall, F1 Score and Accuracy. The R2 score will tell us how well the attributes are predicting our target attribute. Precision is used to see how well the data is classsifying our outcome. For example if we were trying to predict whether or not an e-mail was spam, precision would be used to classify this. Recall is a metric that determines how accurate our precision is. The F1 score combines recall and precision into one metric as their over-all performance. Accuracy is a metric used to tell us how well the entire model is doing at making our predictions.
5. Is there anything else that your director or board of directors need to know about this project?
The dataset is downloaded from UC Irvine's machine learning repository. The data is from two Portugese schools gathered from reports and questionnaires. The data is relevant as there are over 300 student submissions.
Get the Data
1. Correctly import your data (CodeGrade will assume that your data is in the same folder as your notebook just like with your other assignments)

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.base import clone
from scipy import stats
#Above is a list of imports that will be used to perform our calculations.

# Here our data is uploaded from a csv file in excel into our notebook
student_matrix = pd.read_csv("student-mat.csv")
2. *Check the size and type of data

student_matrix.size
# This number counts every individual entry
13825

student_matrix.dtypes
#This tells us the datatype of every attribute. An object is either text or text combined with a numeric value.
#Float64 is a number that includes a decimal up to 64. Int64 is a number without a decimal place up to 64
school          object
sex             object
age            float64
address         object
famsize         object
Pstatus         object
Medu             int64
Fedu             int64
Mjob            object
Fjob            object
reason          object
guardian        object
traveltime       int64
studytime        int64
failures         int64
schoolsup       object
famsup          object
paid            object
activities      object
nursery         object
higher          object
internet        object
romantic        object
famrel           int64
freetime         int64
goout            int64
Dalc             int64
Walc             int64
health           int64
absences_G1    float64
absences_G2    float64
absences_G3    float64
G1               int64
G2               int64
G3               int64
dtype: object
3. List the available features and their data descriptions so that your director/board of directors can understand your work
school - student's school ("GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
sex - student's sex ("F" - female or "M" - male)
age - student's age (numeric from 15 to 22)
address - student's home address type ("U" - urban or "R" - rural)
famsize - family size ("LE3" - less or equal to 3 or "GT3" - greater than 3)
Pstatus - parent's cohabitation status ("T" - living together or "A" - apart)
Medu - mother's education (0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
Fedu - father's education (0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
Mjob - mother's job ( "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
Fjob - father's job ("teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
reason - reason to choose this school (close to "home", school "reputation", "course" preference or "other")
guardian - student's guardian ("mother", "father" or "other")
traveltime - home to school travel time (1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
studytime - weekly study time (1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
failures - number of past class failures (n if 1<=n<3, else 4)
schoolsup - extra educational support (yes or no)
famsup - family educational support (yes or no)
paid - extra paid classes within the course subject (Math or Portuguese) (yes or no)
activities - extra-curricular activities (yes or no)
nursery - attended nursery school (yes or no)
higher - wants to take higher education (yes or no)
internet - Internet access at home (yes or no)
romantic - with a romantic relationship (yes or no)
famrel - quality of family relationships (from 1 - very bad to 5 - excellent)
freetime - free time after school (from 1 - very low to 5 - very high)
goout - going out with friends (from 1 - very low to 5 - very high)
Dalc - workday alcohol consumption (from 1 - very low to 5 - very high)
Walc - weekend alcohol consumption (from 1 - very low to 5 - very high)
health - current health status (from 1 - very bad to 5 - very good)
absences_G1 - number of school absences for G1 term (numeric)
absences_G2 - number of school absences for G2 term (numeric)
absences_G3 - number of school absences for G3 term (numeric)
these grades are related with the course math subject
G1 - first term grade (numeric: from 0 to 20)
G2 - second term grade (numeric: from 0 to 20)
G3 - final grade (numeric: from 0 to 20, ← this is your output target)
4. Identify the target or label attribute
The target attribute is G3 also known as the final grades. This will determine the students that need the most help.

Y = student_matrix["G3"]
X = student_matrix.drop(["G3"], axis=1)
#Here the data is split into the dependent variable (Y), also our target output and the independent variables (X)
converting grades to US equivalent

## converting G3 column to US Grade equivalent
US_GRADE_EQUAL ={
    0: 'F',
    1: 'F',
    2: 'F',
    3: 'F',
    4: 'F',
    5: 'F',
    6: 'F',
    7: 'F',
    8: 'F',
    9: 'F',
    10: 'C',
    11: 'C',
    12: 'C',
    13: 'C',
    14: 'B',
    15: 'B',
    16: 'A',
    17: 'A',
    18: 'A+',
    19: 'A+',
    20: 'A+'
}
​
Y = Y.map(US_GRADE_EQUAL)
​
Y.value_counts()
​
C     165
F     130
B      60
A      22
A+     18
Name: G3, dtype: int64

y_le = LabelEncoder()
Y = y_le.fit_transform(Y)
Y = pd.Series(Y)

y_le.classes_.tolist(), y_le.transform(y_le.classes_.tolist())
(['A', 'A+', 'B', 'C', 'F'], array([0, 1, 2, 3, 4]))
Because we are going with a classification model, the Portugese grades are converted to the Alphabetical equivalent as per the three previous cells

student_matrix.select_dtypes(include="object").columns
#This is a list of attributes that are non-numeric aka categorical
Index(['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic'],
      dtype='object')
5. Correctly split your data into a training and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
#The data is being split into a training set and test set. The Training set is used to train the model while the 
#test set evaluates the models performance.
Explore the Data
1. *Thoroughly study the training set attributes and their characteristics
1.1 Analysing the Target Variable

# The graph shows the distribution of final grades
grades = Y
​
plt.figure(figsize=(8, 6))
plt.hist(grades, bins=10, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('G3 Grades')
plt.ylabel('Frequency')
plt.title('Distribution of G3 Grades')
plt.show()

1.2 Missing Value Analysis

X_train.isna().sum()
school          0
sex             0
age            11
address         0
famsize         0
Pstatus         0
Medu            0
Fedu            0
Mjob            0
Fjob            0
reason          0
guardian        0
traveltime      0
studytime       0
failures        0
schoolsup       0
famsup          0
paid            0
activities      0
nursery         0
higher          0
internet        0
romantic        0
famrel          0
freetime        0
goout           0
Dalc            0
Walc            0
health          0
absences_G1    11
absences_G2    11
absences_G3    11
G1              0
G2              0
dtype: int64
Here we see where there are missing values in the data. There are no columns with a very large missing data so we will remove them
2. *Produce at least four visualizations using your training data to assist in exploring the data. You should ensure that your visuals are informative and visually appealing, not purely using the default plots. Explain what each chart shows, why it’s important, and what insights did you obtain from the plots. (matplotlib and seaborn are the only packages available in CodeGrade)
1.1 Distribution of each variable against Target variable

# The scatter plots below give a visualization of how each attribute correlates to the final grades
for column in X_train.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(y_train, X_train[column], label=f'{column} vs G3_US_EQUAL')
    plt.title(f'Scatter Plot of {column} vs G3_US_EQUAL')
    plt.xlabel("G3_US_EQUAL")
    plt.ylabel(column)
    plt.legend()
    plt.show()


































1.2 Distribution analysis of each variable

#Below is the distribution of each varaiable. This will help us determine outliers and help decide how to go about
#cleaning the data.
plt.figure(figsize=(10, 10))
print(X_train.columns)
for column in X_train.columns:
    plt.hist(student_matrix[column], bins=10, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f"""Distribution of {column} column""")
    plt.show()
Index(['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences_G1', 'absences_G2', 'absences_G3', 'G1',
       'G2'],
      dtype='object')


































3. *Study the correlations between attributes

sns.heatmap(X_train.corr(numeric_only=True), annot=False, cmap='coolwarm')
​
<Axes: >

As seen from the correlation plot, almost all the correlation boxes are approximately blue hence there is a very small amount of direct correlation or connection. We will have to engineer a few features to check

data = X_train.copy()
data["NumericTarget"] = y_train
categorical_columns = data.select_dtypes(include=['object']).columns
​
# Create subplots - setting up the plot grid
fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(8, 6 * len(categorical_columns)))
​
# Ensure axes is an array even if there's only one plot
if len(categorical_columns) == 1:
    axes = [axes]
​
# Loop through categorical columns
for idx, column in enumerate(categorical_columns):
    # Perform ANOVA
    groups = [group['NumericTarget'].values for name, group in data.groupby(column)]
    fvalue, pvalue = stats.f_oneway(*groups)
    print(f"ANOVA results for {column}: F-value = {fvalue}, P-value = {pvalue}")
​
    # Create a box plot
    sns.boxplot(x=column, y='NumericTarget', data=data, ax=axes[idx])
    axes[idx].set_title(f'Box Plot of NumericTarget by {column}\nF-value={fvalue:.2f}, P-value={pvalue:.4f}')
    axes[idx].set_xlabel(column)
    axes[idx].set_ylabel('NumericTarget')
​
# Adjust layout
plt.tight_layout()
plt.show()
​
ANOVA results for school: F-value = 0.9960720406299977, P-value = 0.3190313084941424
ANOVA results for sex: F-value = 1.323975720871076, P-value = 0.2507552731088256
ANOVA results for address: F-value = 0.5158372509213973, P-value = 0.4731568606833193
ANOVA results for famsize: F-value = 0.594665492083526, P-value = 0.4412005386617871
ANOVA results for Pstatus: F-value = 0.9288782531047133, P-value = 0.33589622139028896
ANOVA results for Mjob: F-value = 2.2708677132352717, P-value = 0.06155046104697915
ANOVA results for Fjob: F-value = 1.0012114655397053, P-value = 0.40708415789924146
ANOVA results for reason: F-value = 1.242743095439898, P-value = 0.294282599177183
ANOVA results for guardian: F-value = 1.0156572682472473, P-value = 0.3633547168525363
ANOVA results for schoolsup: F-value = 5.737732972644318, P-value = 0.017189645843974862
ANOVA results for famsup: F-value = 0.0054825469211609605, P-value = 0.941022343863122
ANOVA results for paid: F-value = 1.097911299896966, P-value = 0.2955307983190281
ANOVA results for activities: F-value = 0.10845603015613638, P-value = 0.7421283781851122
ANOVA results for nursery: F-value = 0.4922132574919917, P-value = 0.48346133010374115
ANOVA results for higher: F-value = 10.855080440327852, P-value = 0.001097896248864885
ANOVA results for internet: F-value = 5.313169285986579, P-value = 0.02181666981158117
ANOVA results for romantic: F-value = 1.4326782074035314, P-value = 0.2322307405420646

Above an Anova test was performed and then graphed for visual interpretation. The F and P values were interpretted to see which groups are more connected to the target attribute. The higher the F value, the more signicance it is in correlation. The lower the P value the less likely the information registered was by chance.
Prepare the Data
1. Based on your exploration of the data above, perform feature selection to narrow down your data. (While not required, we would suggest that you create a function or custom transformer to handle this step so that you can more easily transform your test data.)
Based on the above findings the data, feature selection will be performed which will decide which are the most relevant features to train our model on. Using custom transformers, this will help reduce the dimentionality or range of our data so that the model will run easier and helps get rid of "noisy", irrelevant, or redundant data.

student_matrix.famsize.unique()
array(['GT3', 'LE3'], dtype=object)

#write a sklearn transformer to drop na values
class DropNaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
​
    def transform(self, X):
        print("Transforming DropNaTransformer")
        return X.dropna()
​
​
test_df = pd.DataFrame({"a":[0, np.NaN], "b":[1, 2]})
print(test_df)
transformer = DropNaTransformer()
transformed_df = transformer.transform(test_df)
print("After droppping na values:")
print(transformed_df)
​
     a  b
0  0.0  1
1  NaN  2
Transforming DropNaTransformer
After droppping na values:
     a  b
0  0.0  1

class AbsencesFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, remove=False):
        # Initialize the transformer with the option to remove the original columns
        self.remove = remove
​
    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data,
        # so the fit method just returns self.
        return self
​
    def transform(self, X):
        # Check if input is a DataFrame
        print("Transforming AbsencesFeatureGenerator")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Ensure the necessary columns are in the DataFrame
        required_columns = ['absences_G1', 'absences_G2', 'absences_G3']
        if not all(col in X.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Create the new 'absences' column by summing the specified columns
        X = X.copy()  # Create a copy of the DataFrame to avoid changing the original data
        X['absences'] = X['absences_G1'] + X['absences_G2'] + X['absences_G3']
        
        # Remove the original columns if requested
        if self.remove:
            X.drop(['absences_G1', 'absences_G2', 'absences_G3'], axis=1, inplace=True)
        
        return X
​
test_df = pd.DataFrame({"absences_G1":[0, 1], "absences_G2":[1, 2], "absences_G3":[2,3] })
print(test_df)
transformer = AbsencesFeatureGenerator(remove=True)
transformed_df = transformer.transform(test_df)
print("After adding absences column:")
print(transformed_df)
   absences_G1  absences_G2  absences_G3
0            0            1            2
1            1            2            3
Transforming AbsencesFeatureGenerator
After adding absences column:
   absences
0         3
1         6

class FailuresCategorizer(TransformerMixin, BaseEstimator):
    def __init__(self, remove=False):
        # Initialize with the option to remove the original 'failures' column
        self.remove = remove
​
    def fit(self, X, y=None):
        # No fitting process needed for this transformer, return self to allow chaining
        return self
​
    def transform(self, X):
        # Check if the dataframe contains the 'failures' column
        print("Transforming FailuresCategorizer")
        if 'failures' not in X.columns:
            raise ValueError("DataFrame must contain a 'failures' column")
        
        # Create a new column based on the 'failures' column
        X = X.copy()  # Make a copy to avoid changing the original data
        X['failures_cat'] = X['failures'].apply(lambda x: 1 if x > 0 else 0)
        
        if self.remove:
            # Remove the 'failures' column if specified
            X.drop(['failures'], axis=1, inplace=True)
​
        return X
​
test_df = pd.DataFrame({'failures': [0, 1, 2, 0, 3]})
print(test_df)
transformer = FailuresCategorizer(remove=True)
transformed_df = transformer.transform(test_df)
print("After adding failures_cat column:")
print(transformed_df)
   failures
0         0
1         1
2         2
3         0
4         3
Transforming FailuresCategorizer
After adding failures_cat column:
   failures_cat
0             0
1             1
2             1
3             0
4             1

class BaseFeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, remove=False, name='BaseFeatureTransformer'):
        self.remove = remove
        self.name = name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Make a copy to avoid changing the original DataFrame
​
        print(f"Transforming {self.name}")
        if self.column_name not in X.columns:
            raise ValueError(f"DataFrame must contain the column '{self.column_name}'")
        
        X[self.new_column_name] = X[self.column_name].apply(self.category_function)
        if self.remove:
            X.drop([self.column_name], axis=1, inplace=True)
        
        return X
​

class AlcoholWeekendTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='AlcoholWeekendTransformer')
        self.column_name = 'Walc'
        self.new_column_name = 'Walc_cat'
        self.category_function = lambda walc: 0 if walc < 1.5 else (1 if walc < 3.5 else 2)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[1], line 1
----> 1 class AlcoholWeekendTransformer(BaseFeatureTransformer):
      2     def __init__(self, remove=False):
      3         super().__init__(remove, name='AlcoholWeekendTransformer')

NameError: name 'BaseFeatureTransformer' is not defined


df = pd.DataFrame({'Walc': [0, 2, 4]})
transformer = AlcoholWeekendTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'Walc_cat': [0, 1, 2]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming AlcoholWeekendTransformer

class HealthCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False, name='HealthCategoryTransformer'):
        super().__init__(remove)
        self.column_name = 'health'
        self.new_column_name = 'health_cat'
        self.category_function = self.get_health_category
​
    def get_health_category(self, health):
        if health < 2.5:
            return 0
        elif health < 4.5:
            return 1
        else:
            return 2
        
​
df = pd.DataFrame({'health': [2, 5, 10]})
transformer = HealthCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'health_cat': [0, 2, 2]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming BaseFeatureTransformer

class DailyAlcoholTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='DailyAlcoholTransformer')
        self.column_name = 'Dalc'
        self.new_column_name = 'Dalc_cat'
        self.category_function = lambda dalc: 0 if dalc < 1.5 else 1
​
df = pd.DataFrame({'Dalc': [0, 1, 3]})
transformer = DailyAlcoholTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'Dalc_cat': [0, 0, 1]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming DailyAlcoholTransformer

class GooutTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='GooutTransformer')
        self.column_name = 'goout'
        self.new_column_name = 'goout_cat'
        self.category_function = self.get_goout_category
    
    def get_goout_category(self,goout):
        if goout < 1.5:
            return 0
        elif goout < 3.5:
            return 2
        else:
            return 1
​
df = pd.DataFrame({'goout': [0, 3, 10]})
transformer = GooutTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'goout_cat': [0, 2, 1]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming GooutTransformer

class FreeTimeCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='FreeTimeCategoryTransformer')
        self.column_name = 'freetime'
        self.new_column_name = 'freetime_cat'
        self.category_function = self.get_freetime_category
    
    def get_freetime_category(self, freetime):
        if freetime < 1.5:
            return 0
        elif freetime < 4.5:
            return 1
        else:
            return 2
​
df = pd.DataFrame({'freetime': [0, 4, 10]})
transformer = FreeTimeCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'freetime_cat': [0, 1, 2]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming FreeTimeCategoryTransformer

class FamilyRelativeTransormer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='FamilyRelativeTransormer')
        self.column_name = 'famrel'
        self.new_column_name = 'famrel_cat'
        self.category_function = self.get_famrel_category
    
    def get_famrel_category(self, famrel):
        if famrel <= 3:
            return 0
        else:
            return 1
​
df = pd.DataFrame({'famrel': [0, 4, 3]})
transformer = FamilyRelativeTransormer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'famrel_cat': [0, 1, 0]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming FamilyRelativeTransormer

class AgeCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='AgeCategoryTransformer')
        self.column_name = 'age'
        self.new_column_name = 'age_cat'
        self.category_function = self.get_age_category
    
    def get_age_category(self, age):
        if age <= 17:
            return 0
        elif age < 20:
            return 1
        else:
            return 2
​
df = pd.DataFrame({'age': [17, 19, 20]})
transformer = AgeCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'age_cat': [0, 1, 2]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming AgeCategoryTransformer

class TravelTimeCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='TravelTimeCategoryTransformer')
        self.column_name = 'traveltime'
        self.new_column_name = 'traveltime_cat'
        self.category_function = self.get_traveltime_category
    
    def get_traveltime_category(self, traveltime):
        if traveltime < 1.5:
            return 0
        else:
            return 1
​
df = pd.DataFrame({'traveltime': [0, 3, 1]})
transformer = TravelTimeCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'traveltime_cat': [0, 1, 0]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming TravelTimeCategoryTransformer

class EduCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, column_name='fedu', remove=False):
        super().__init__(remove, name='EduCategoryTransformer')
        self.column_name = column_name
        self.new_column_name = f"""{self.column_name}_cat"""
        self.category_function = self.get_medu_fedu_category
    
    def get_medu_fedu_category(self, medu):
        if medu <=2:
            return 0
        elif medu <=3:
            return 1
        else:
            return 2
​
df = pd.DataFrame({'fedu': [0, 2, 3, 4]})
transformer = EduCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'fedu_cat': [0, 0, 1, 2]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming EduCategoryTransformer

class StudyTimeCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='StudyTimeCategoryTransformer')
        self.column_name = 'studytime'
        self.new_column_name = 'studytime_cat'
        self.category_function = self.get_studytime_category
    
    def get_studytime_category(self, studytime):
        if studytime <=3:
            return 0
        elif studytime <=5:
            return 1
        else:
            return 2   
​
df = pd.DataFrame({'studytime': [0, 4, 6]})
transformer = StudyTimeCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'studytime_cat': [0, 1, 2]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming StudyTimeCategoryTransformer

class FamsizeCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='FamsizeCategoryTransformer')
        self.column_name = 'famsize'
        self.new_column_name = 'famsize_cat'
        self.category_function = self.get_famsize_category
    
    def get_famsize_category(self, famsize):
        if famsize == 'LE3':
            return 0
        else:
            return 1
​
df = pd.DataFrame({'famsize': ['GT3', 'LE3', 'GT3']})
transformer = FamsizeCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'famsize_cat': [1, 0, 1]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming FamsizeCategoryTransformer

class MJobCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='MJobCategoryTransformer')
        self.column_name = 'Mjob'
        self.new_column_name = 'Mjob_cat'
        self.category_function = self.get_mjob_category
    
    def get_mjob_category(self, mjob):
        if mjob in ['at_home', 'teacher', 'health']:
            return 0
        else:
            return 1
​
df = pd.DataFrame({'Mjob': ['other', 'services', 'other', 'at_home']})
transformer = MJobCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'Mjob_cat': [1, 1, 1, 0]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming MJobCategoryTransformer

class FJobCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='FJobCategoryTransformer')
        self.column_name = 'Fjob'
        self.new_column_name = 'Fjob_cat'
        self.category_function = self.get_fjob_category
    
    def get_fjob_category(self, fjob):
        if fjob in ['at_home', 'teacher', 'health']:
            return 0
        else:
            return 1
​
df = pd.DataFrame({'Fjob': ['other', 'services', 'other', 'at_home']})
transformer = FJobCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'Fjob_cat': [1, 1, 1, 0]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming FJobCategoryTransformer

class GuardianCategoryTransformer(BaseFeatureTransformer):
    def __init__(self, remove=False):
        super().__init__(remove, name='GuardianCategoryTransformer')
        self.column_name = 'guardian'
        self.new_column_name = 'guardian_cat'
        self.category_function = self.get_guardian_category
    
    def get_guardian_category(self, guardian):
        if guardian in ['mother']:
            return 0
        else:
            return 1
​
df = pd.DataFrame({'guardian': ['mother', 'father', 'mother', 'father']})
transformer = GuardianCategoryTransformer(remove=True)
transformed_df = transformer.transform(df)
expected_output = pd.DataFrame({'guardian_cat': [0, 1, 0, 1]})
pd.testing.assert_frame_equal(transformed_df, expected_output)
Transforming GuardianCategoryTransformer
2. Create at least one data pipeline to handle the data preparation steps

X_train.isna().sum()
school          0
sex             0
age            11
address         0
famsize         0
Pstatus         0
Medu            0
Fedu            0
Mjob            0
Fjob            0
reason          0
guardian        0
traveltime      0
studytime       0
failures        0
schoolsup       0
famsup          0
paid            0
activities      0
nursery         0
higher          0
internet        0
romantic        0
famrel          0
freetime        0
goout           0
Dalc            0
Walc            0
health          0
absences_G1    11
absences_G2    11
absences_G3    11
G1              0
G2              0
dtype: int64

# A pipeline is made to streamline the transformers into the data making the model more efficient and scalable.
​
pipeline = Pipeline([
    ('failures', FailuresCategorizer(remove=True)),
    ('age_cat', AgeCategoryTransformer(remove=True)),
    ('traveltime_cat', TravelTimeCategoryTransformer(remove=True)),
    ('famrel_cat', FamilyRelativeTransormer(remove=True)),
    ('freetime_cat', FreeTimeCategoryTransformer(remove=True)),
    ('goout_cat', GooutTransformer(remove=True)),
    ('Dalc_cat', DailyAlcoholTransformer(remove=True)),
    ('health_cat', HealthCategoryTransformer(remove=True)),
    ('edu_medu_cat', EduCategoryTransformer(column_name='Medu', remove=True)),
    ('edu_fedu_cat', EduCategoryTransformer(column_name='Fedu', remove=True)),
    ('studytime_cat', StudyTimeCategoryTransformer(remove=True)),
    ('famsize_cat', FamsizeCategoryTransformer(remove=True)),
    ('mjob_cat', MJobCategoryTransformer(remove=True)),
    ('fjob_cat', FJobCategoryTransformer(remove=True)),
    ('guardian_cat', GuardianCategoryTransformer(remove=True)),
])
3. Fill in missing values or drop the rows or columns with missing values inside your pipeline

pipeline.steps.append(('na_removal', DropNaTransformer()))
4. Create a custom transformer in your pipeline that:
● creates a new column in the data that sums the absences_G1, absences_G2, and absences_G3 data and then drops those three columns.
● has a parameter that when equal to True, drops the G1 and G2 columns, and when False, leaves the columns in the data

pipeline.steps.append(('absences', AbsencesFeatureGenerator(remove=True)))

class G1G2RemovalTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, remove=False):
        self.remove = remove
        self.columns = ['G1', 'G2']
​
    def fit(self, X, y=None):
        return self
​
    def transform(self, X):
        X = X.copy()
        if self.remove:
            X = X.drop(self.columns, axis=1)
        return X    
​
testing_frame  = pd.DataFrame({
    'G1': [1, 2, 3],
    'G2': [4, 5, 6],
    'G3': [7, 8, 9],
})
​
transformer = G1G2RemovalTransformer(remove=True)
transformed_frame = transformer.fit_transform(testing_frame)
print(transformed_frame)
​
# we will use this transformer at the very end in the pipeline to remove G1 and G2
   G3
0   7
1   8
2   9
5. Perform feature scaling on continuous numeric data in your pipeline

#Feature scaling is used on our numeric data so they may be on a similar scale. 
#This makes it easier for the features to calculate. 
class MinMaxScalerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.numeric_columns = None
        self.scaler = MinMaxScaler()
​
    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include='number').columns
        self.scaler.fit(X[self.numeric_columns])
        return self
​
    def transform(self, X):
        X = X.copy()
        print("Transforming MinMaxScalerTransformer")
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X
​
​
​
testing_frame = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9],
})
​
transformer = MinMaxScalerTransformer()
transformed_frame = transformer.fit_transform(testing_frame)
print(transformed_frame)
​
pipeline.steps.append(('scaler', MinMaxScalerTransformer()))
​
test_df = pipeline.fit_transform(X_train)
Transforming MinMaxScalerTransformer
     a    b    c
0  0.0  0.0  0.0
1  0.5  0.5  0.5
2  1.0  1.0  1.0
Transforming FailuresCategorizer
Transforming AgeCategoryTransformer
Transforming TravelTimeCategoryTransformer
Transforming FamilyRelativeTransormer
Transforming FreeTimeCategoryTransformer
Transforming GooutTransformer
Transforming DailyAlcoholTransformer
Transforming BaseFeatureTransformer
Transforming EduCategoryTransformer
Transforming EduCategoryTransformer
Transforming StudyTimeCategoryTransformer
Transforming FamsizeCategoryTransformer
Transforming MJobCategoryTransformer
Transforming FJobCategoryTransformer
Transforming GuardianCategoryTransformer
Transforming DropNaTransformer
Transforming AbsencesFeatureGenerator
Transforming MinMaxScalerTransformer
6. Ordinal encode features that are either binary or that are ordinal in nature; and/or one-hot encode nominal or categorical data in a pipeline

#OneHotEncoding turns our categorical data with two values into a binary value system or the number 1 and 0.
class OneHotEncodingTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, remove=False):
        self.remove = remove
        self.encoders = None
​
    def fit(self, X, y=None):
        self.columns = ["health_cat","Dalc_cat","goout_cat","freetime_cat","famrel_cat"]
        self.encoders ={col: OneHotEncoder(handle_unknown='ignore') for col in self.columns}
        # Fitting the encoders
        for col in self.encoders:
            self.encoders[col].fit(X[[col]])
        return self
​
    def transform(self, X):
        X = X.copy()
        print("Transforming OneHotEncodingTransformer")
​
        # Transforming the data
        for col in self.columns:
            encoded = self.encoders[col].transform(X[[col]]).toarray()
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0]], index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            
        return X
​
​
#OneHotEncodingTransformer(remove=True).fit_transform(test_df)["Dalc_cat_0.0"].value_counts()
​
# ohe = OneHotEncoder().fit(test_df[["Dalc_cat"]])
# ohe.transform(test_df[["Dalc_cat"]]).toarray()
pipeline.steps.append(('one_hot_encoder', OneHotEncodingTransformer(remove=True)))

test_df.Dalc_cat.value_counts()
0.0    209
1.0     96
Name: Dalc_cat, dtype: int64

X_train
school	sex	age	address	famsize	Pstatus	Medu	Fedu	Mjob	Fjob	...	freetime	goout	Dalc	Walc	health	absences_G1	absences_G2	absences_G3	G1	G2
181	GP	M	16.0	U	GT3	T	3	3	services	other	...	2	3	1	2	3	0.0	0.0	2.0	12	13
194	GP	M	16.0	U	GT3	T	2	3	other	other	...	3	3	1	1	3	0.0	0.0	0.0	13	14
173	GP	F	16.0	U	GT3	T	1	3	at_home	services	...	3	5	1	1	3	0.0	0.0	0.0	8	7
63	GP	F	16.0	U	GT3	T	4	3	teacher	health	...	4	4	2	4	4	0.0	0.0	2.0	10	9
253	GP	M	16.0	R	GT3	T	2	1	other	other	...	3	2	1	3	3	0.0	0.0	0.0	8	9
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
71	GP	M	15.0	U	GT3	T	4	2	other	other	...	3	3	1	1	3	0.0	0.0	0.0	10	10
106	GP	F	15.0	U	GT3	T	2	2	other	other	...	1	2	1	1	3	NaN	NaN	NaN	7	8
270	GP	F	19.0	U	GT3	T	3	3	other	services	...	3	5	3	3	5	2.0	2.0	11.0	9	9
348	GP	F	17.0	U	GT3	T	4	3	health	other	...	4	3	1	3	4	0.0	0.0	0.0	13	15
102	GP	M	15.0	U	GT3	T	4	4	services	other	...	3	3	1	1	5	0.0	0.0	4.0	10	13
316 rows × 34 columns
7. Create a Column Transformer to transform your numeric and categorical data

#create a transformer which converts only categorical columns to numerical
class CategoricalToNumericalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Get the categorical columns
        self.categorical_columns_ = X.select_dtypes(include='object').columns
        self.le_dict = {}
        for col in self.categorical_columns_:
            le = LabelEncoder()
            le = le.fit(X[col])
            self.le_dict[col] = le
        return self
​
    def transform(self, X):
        # Apply LabelEncoder to categorical columns
        print("Transforming CategoricalToNumericalTransformer")
        X_cat = X.copy()
        for col in self.categorical_columns_:
            X_cat[col] = self.le_dict[col].transform(X_cat[col])
        return X_cat
​
# Sample data
testing_frame = pd.DataFrame({
    'school': ['GP', 'GP', 'MS', 'GP', 'GP']
})
​
# Creating and using the transformer
transformer = CategoricalToNumericalTransformer()
transformed_frame = transformer.fit_transform(testing_frame)
print(transformed_frame)
​
pipeline.steps.append(('cat_to_num_encoder', CategoricalToNumericalTransformer()))
Transforming CategoricalToNumericalTransformer
   school
0       0
1       0
2       1
3       0
4       0
8. Correctly transform your training data using the above data preparation steps and pipelines. You should have two distinct sets of transformed training data: one containing the G1/G2 columns and another without the G1/G2 columns.

pipe_no_g1_g2 = clone(pipeline)
pipe_no_g1_g2.steps.append(('g1_g2_removal', G1G2RemovalTransformer(remove=True)))
​
X1_train_encoded = pipeline.fit_transform(X_train)
X1_not_G1_G2 = pipe_no_g1_g2.fit_transform(X_train)
​
y_train_clean = y_train.loc[X1_train_encoded.index]
​
Transforming FailuresCategorizer
Transforming AgeCategoryTransformer
Transforming TravelTimeCategoryTransformer
Transforming FamilyRelativeTransormer
Transforming FreeTimeCategoryTransformer
Transforming GooutTransformer
Transforming DailyAlcoholTransformer
Transforming BaseFeatureTransformer
Transforming EduCategoryTransformer
Transforming EduCategoryTransformer
Transforming StudyTimeCategoryTransformer
Transforming FamsizeCategoryTransformer
Transforming MJobCategoryTransformer
Transforming FJobCategoryTransformer
Transforming GuardianCategoryTransformer
Transforming DropNaTransformer
Transforming AbsencesFeatureGenerator
Transforming MinMaxScalerTransformer
Transforming OneHotEncodingTransformer
Transforming CategoricalToNumericalTransformer
Transforming FailuresCategorizer
Transforming AgeCategoryTransformer
Transforming TravelTimeCategoryTransformer
Transforming FamilyRelativeTransormer
Transforming FreeTimeCategoryTransformer
Transforming GooutTransformer
Transforming DailyAlcoholTransformer
Transforming BaseFeatureTransformer
Transforming EduCategoryTransformer
Transforming EduCategoryTransformer
Transforming StudyTimeCategoryTransformer
Transforming FamsizeCategoryTransformer
Transforming MJobCategoryTransformer
Transforming FJobCategoryTransformer
Transforming GuardianCategoryTransformer
Transforming DropNaTransformer
Transforming AbsencesFeatureGenerator
Transforming MinMaxScalerTransformer
Transforming OneHotEncodingTransformer
Transforming CategoricalToNumericalTransformer
9. *Output the shape of your two transformed training sets to show your custom transformer correctly removed the two columns

print(X1_train_encoded.shape)
print(X1_not_G1_G2.shape)
(305, 45)
(305, 43)
Shortlist Promising Models
1. Fit three or more promising models to your data using your transformed data
The models being used are best use for Classification.

y_train_clean = y_train.loc[X1_train_encoded.index]

randomForestNotG1G2 = RandomForestClassifier()
randomForestNotG1G2.fit(X1_not_G1_G2, y_train_clean)
​
randomForest = RandomForestClassifier()
randomForest.fit(X1_train_encoded, y_train_clean)
​
print(randomForestNotG1G2.score(X1_not_G1_G2, y_train_clean))
print(randomForest.score(X1_train_encoded, y_train_clean))
​
​
## testing
​
X_test_encoded = pipeline.transform(X_test)
y_test_clean = y_test.loc[X_test_encoded.index]
X_test_not_G1_G2 = X_test_encoded.drop(["G1","G2"], axis=1)
​
accuracy_rf = accuracy_score(y_test_clean, randomForest.predict(X_test_encoded))
print("Accuracy", accuracy_rf)
precision_rf = precision_score(y_test_clean, randomForest.predict(X_test_encoded), average="macro")
print("Precision", precision_rf)
recall_rf = recall_score(y_test_clean, randomForest.predict(X_test_encoded), average="macro")
print("Recall", recall_rf)
f1_rf = f1_score(y_test_clean, randomForest.predict(X_test_encoded) , average="macro")
print("F1", f1_rf)
r2_score_rf = r2_score(y_test_clean, randomForest.predict(X_test_encoded))
print("r2_score", r2_score_rf)
​
​
accuracy_rf_notG1G2 = accuracy_score(y_test_clean, randomForestNotG1G2.predict(X_test_not_G1_G2))
print("Accuracy", accuracy_rf_notG1G2)
precision_rf_notG1G2 = precision_score(y_test_clean, randomForestNotG1G2.predict(X_test_not_G1_G2), average="macro")
print("Precision", precision_rf_notG1G2)
recall_rf_notG1G2 = recall_score(y_test_clean, randomForestNotG1G2.predict(X_test_not_G1_G2), average="macro")
print("Recall", recall_rf_notG1G2)
f1_rf_notG1G2 = f1_score(y_test_clean, randomForestNotG1G2.predict(X_test_not_G1_G2) , average="macro")
print("F1", f1_rf_notG1G2)
r2_score_rf_notG1G2 = r2_score(y_test_clean, randomForestNotG1G2.predict(X_test_not_G1_G2))
print("r2_score", r2_score_rf_notG1G2)
1.0
1.0
Transforming FailuresCategorizer
Transforming AgeCategoryTransformer
Transforming TravelTimeCategoryTransformer
Transforming FamilyRelativeTransormer
Transforming FreeTimeCategoryTransformer
Transforming GooutTransformer
Transforming DailyAlcoholTransformer
Transforming BaseFeatureTransformer
Transforming EduCategoryTransformer
Transforming EduCategoryTransformer
Transforming StudyTimeCategoryTransformer
Transforming FamsizeCategoryTransformer
Transforming MJobCategoryTransformer
Transforming FJobCategoryTransformer
Transforming GuardianCategoryTransformer
Transforming DropNaTransformer
Transforming AbsencesFeatureGenerator
Transforming MinMaxScalerTransformer
Transforming OneHotEncodingTransformer
Transforming CategoricalToNumericalTransformer
Accuracy 0.7631578947368421
Precision 0.6457671957671958
Recall 0.694
F1 0.6686075300947665
r2_score 0.6062176165803109
Accuracy 0.3684210526315789
Precision 0.23578431372549016
Recall 0.2375
F1 0.21566416040100256
r2_score -0.08290155440414493
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))

decisionTree = DecisionTreeClassifier()
​
decisionTree.fit(X1_train_encoded, y_train_clean)
​
​
decisionTreeNotG1G2 = DecisionTreeClassifier()
decisionTreeNotG1G2.fit(X1_not_G1_G2, y_train_clean)
​
## testing
​
accuracy_dt = accuracy_score(y_test_clean, decisionTree.predict(X_test_encoded))
​
print(accuracy_dt)
precision_dt = precision_score(y_test_clean, decisionTree.predict(X_test_encoded), average="macro")
​
print(precision_dt)
​
recall_dt = recall_score(y_test_clean, decisionTree.predict(X_test_encoded), average="macro")
print(recall_dt)
​
f1_dt = f1_score(y_test_clean, decisionTree.predict(X_test_encoded) , average="macro")
print(f1_dt)
​
r2_dt = r2_score(y_test_clean, decisionTree.predict(X_test_encoded))
print(r2_dt)
​
accuracy_dt_notG1G2 = accuracy_score(y_test_clean, decisionTreeNotG1G2.predict(X_test_not_G1_G2))
​
print(accuracy_dt_notG1G2)
precision_dt_notG1G2 = precision_score(y_test_clean, decisionTreeNotG1G2.predict(X_test_not_G1_G2), average="macro")
​
print(precision_dt_notG1G2)
​
recall_dt_notG1G2 = recall_score(y_test_clean, decisionTreeNotG1G2.predict(X_test_not_G1_G2), average="macro")
print(recall_dt_notG1G2)
​
f1_dt_notG1G2 = f1_score(y_test_clean, decisionTreeNotG1G2.predict(X_test_not_G1_G2) , average="macro")
print(f1_dt_notG1G2)
​
r2_dt_notG1G2 = r2_score(y_test_clean, decisionTreeNotG1G2.predict(X_test_not_G1_G2))
print(r2_dt_notG1G2)
​
0.6842105263157895
0.5866459627329192
0.6530714285714285
0.5997529200359389
0.49882242110221386
0.2631578947368421
0.22379934879934876
0.24823809523809523
0.22304586918548233
-1.52378709373528

XGBoost = GradientBoostingClassifier()
​
XGBoost.fit(X1_train_encoded, y_train_clean)
​
XGBoostNotG1G2 = GradientBoostingClassifier()
XGBoostNotG1G2.fit(X1_not_G1_G2, y_train_clean)
​
## testing
​
accuracy_xgb = accuracy_score(y_test_clean, XGBoost.predict(X_test_encoded))
​
print(accuracy_xgb)
precision_xgb = precision_score(y_test_clean, XGBoost.predict(X_test_encoded), average="macro")
​
print(precision_xgb)
​
recall_xgb = recall_score(y_test_clean, XGBoost.predict(X_test_encoded), average="macro")
print(recall_xgb)
​
f1_xgb = f1_score(y_test_clean, XGBoost.predict(X_test_encoded) , average="macro")
print(f1_xgb)
​
r2_xgb = r2_score(y_test_clean, XGBoost.predict(X_test_encoded))
print(r2_xgb)
​
print("-------------------")
accuracy_xgb_notG1G2 = accuracy_score(y_test_clean, XGBoostNotG1G2.predict(X_test_not_G1_G2))
​
print(accuracy_xgb_notG1G2)
precision_xgb_notG1G2 = precision_score(y_test_clean, XGBoostNotG1G2.predict(X_test_not_G1_G2), average="macro")
​
print(precision_xgb_notG1G2)
​
recall_xgb_notG1G2 = recall_score(y_test_clean, XGBoostNotG1G2.predict(X_test_not_G1_G2), average="macro")
print(recall_xgb_notG1G2)
​
f1_xgb_notG1G2 = f1_score(y_test_clean, XGBoostNotG1G2.predict(X_test_not_G1_G2) , average="macro")
print(f1_xgb_notG1G2)
​
r2_xgb_notG1G2 = r2_score(y_test_clean, XGBoostNotG1G2.predict(X_test_not_G1_G2))
print(r2_xgb_notG1G2)
0.7763157894736842
0.6818165815877487
0.7315714285714285
0.6686601307189541
0.7404616109279322
-------------------
0.42105263157894735
0.3284878863826232
0.3823333333333333
0.3319254658385093
0.051342439943476315
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))

lor = LogisticRegression()
​
lor.fit(X1_train_encoded, y_train_clean)
​
lorNotG1G2 = LogisticRegression()
lorNotG1G2.fit(X1_not_G1_G2, y_train_clean)
​
## testing
​
accuracy_lor = accuracy_score(y_test_clean, lor.predict(X_test_encoded))
​
print(accuracy_lor)
precision_lor = precision_score(y_test_clean, lor.predict(X_test_encoded), average="macro")
​
print(precision_lor)
​
recall_lor = recall_score(y_test_clean, lor.predict(X_test_encoded), average="macro")
print(recall_lor)
​
f1_lor = f1_score(y_test_clean, lor.predict(X_test_encoded) , average="macro")
print(f1_lor)
​
r2_lor = r2_score(y_test_clean, lor.predict(X_test_encoded))
print(r2_lor)
​
print("-------------------")
accuracy_lor_notG1G2 = accuracy_score(y_test_clean, lorNotG1G2.predict(X_test_not_G1_G2))
​
print(accuracy_lor_notG1G2)
precision_lor_notG1G2 = precision_score(y_test_clean, lorNotG1G2.predict(X_test_not_G1_G2), average="macro")
​
print(precision_lor_notG1G2)
​
recall_lor_notG1G2 = recall_score(y_test_clean, lorNotG1G2.predict(X_test_not_G1_G2), average="macro")
print(recall_xgb_notG1G2)
​
f1_lor_notG1G2 = f1_score(y_test_clean, lorNotG1G2.predict(X_test_not_G1_G2) , average="macro")
print(f1_xgb_notG1G2)
​
r2_lor_notG1G2 = r2_score(y_test_clean, lorNotG1G2.predict(X_test_not_G1_G2))
print(r2_lor_notG1G2)
#This model shows that we get a better statistical outcome if attributes G1 and G2 are used.
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
0.618421052631579
0.6053695324283559
0.5995714285714285
0.5614979505926664
0.4182760244936411
-------------------
0.3815789473684211
0.3811654135338346
0.3823333333333333
0.3319254658385093
0.04239284032030155
2. Compare all three models both with and without the G1/G2 columns with crossvalidation. Output your results.

#perform crossvalidation to check for overfitting and how the models perform. 
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression()
}
​
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'r2_score': make_scorer(r2_score),
           'f1_score': make_scorer(f1_score, average='macro')}
​
results = {}
​
for name, model in models.items():
    result = cross_validate(model, X1_train_encoded, y_train_clean, cv=5, scoring=scoring)
    print(result)
    results[name] = result
    print(f"{name}: Mean Accuracy: {np.mean(result['test_accuracy']):.3f}, Mean F1 Score: {np.mean(result['test_f1_score']):.3f}")
​
​
# Set up the matplotlib figure and axes
fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharey=True)
ax[0].set_title('Comparison of Accuracy')
ax[1].set_title('Comparison of Precision')
ax[2].set_title('Comparison of Recall')
ax[3].set_title('Comparison of R2 Score')
ax[4].set_title('Comparison of F1 Score')
​
index=0
# Plotting
for idx, metric in enumerate(['test_accuracy', 'test_precision', 'test_recall', 'test_r2_score', 'test_f1_score']):
    for name in results:
        ax[idx].bar(name, np.mean(results[name][metric]), label=f"{name} {metric.split('_')[1]}")
    ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
    ax[idx].set_ylabel('Score')
    index+=1
​
plt.legend()
plt.tight_layout()
plt.show()
​
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
{'fit_time': array([0.33025384, 0.32005215, 0.31787324, 0.31118107, 0.29411721]), 'score_time': array([0.0290103 , 0.04070807, 0.03936291, 0.03830099, 0.03703785]), 'test_accuracy': array([0.75409836, 0.75409836, 0.78688525, 0.75409836, 0.63934426]), 'test_precision': array([0.42090909, 0.44363636, 0.60159933, 0.6283871 , 0.38333333]), 'test_recall': array([0.46783626, 0.49296296, 0.66796296, 0.51247863, 0.40034188]), 'test_r2_score': array([0.63958641, 0.40166748, 0.71579205, 0.48330106, 0.45377541]), 'test_f1_score': array([0.44278003, 0.46699101, 0.63082011, 0.53552116, 0.38637993])}
Random Forest: Mean Accuracy: 0.738, Mean F1 Score: 0.492
{'fit_time': array([1.20962095, 1.06314802, 1.00256324, 0.83346891, 0.85749388]), 'score_time': array([0.01892209, 0.01110673, 0.01121306, 0.01324201, 0.011765  ]), 'test_accuracy': array([0.80327869, 0.72131148, 0.7704918 , 0.7704918 , 0.80327869]), 'test_precision': array([0.74179894, 0.66142857, 0.59974747, 0.63628594, 0.8297619 ]), 'test_recall': array([0.72709552, 0.56148148, 0.66055556, 0.65264957, 0.79128205]), 'test_r2_score': array([0.63958641, 0.61108386, 0.61108386, 0.61616651, 0.73426912]), 'test_f1_score': array([0.73203704, 0.57548766, 0.62521008, 0.64407814, 0.80440613])}
Gradient Boosting: Mean Accuracy: 0.774, Mean F1 Score: 0.676
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
{'fit_time': array([0.03856087, 0.03247309, 0.03558397, 0.03664422, 0.05600309]), 'score_time': array([0.01866317, 0.01027703, 0.01031303, 0.01044083, 0.01519084]), 'test_accuracy': array([0.60655738, 0.55737705, 0.63934426, 0.59016393, 0.54098361]), 'test_precision': array([0.33070175, 0.35467492, 0.34247649, 0.5241285 , 0.3131063 ]), 'test_recall': array([0.34736842, 0.33314815, 0.36814815, 0.39957265, 0.29606838]), 'test_r2_score': array([0.30920729, 0.22216773, 0.50637567, 0.18804453, 0.1732817 ]), 'test_f1_score': array([0.33403509, 0.33430353, 0.3547619 , 0.42588005, 0.28806306])}
Logistic Regression: Mean Accuracy: 0.587, Mean F1 Score: 0.347
/var/folders/7v/8544rkhj2vn_rd_7t5dmcbk80000gn/T/ipykernel_96651/358668927.py:36: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax[idx].set_xticklabels(labels=models.keys(), rotation=45)


results
{'Random Forest': {'fit_time': array([0.33025384, 0.32005215, 0.31787324, 0.31118107, 0.29411721]),
  'score_time': array([0.0290103 , 0.04070807, 0.03936291, 0.03830099, 0.03703785]),
  'test_accuracy': array([0.75409836, 0.75409836, 0.78688525, 0.75409836, 0.63934426]),
  'test_precision': array([0.42090909, 0.44363636, 0.60159933, 0.6283871 , 0.38333333]),
  'test_recall': array([0.46783626, 0.49296296, 0.66796296, 0.51247863, 0.40034188]),
  'test_r2_score': array([0.63958641, 0.40166748, 0.71579205, 0.48330106, 0.45377541]),
  'test_f1_score': array([0.44278003, 0.46699101, 0.63082011, 0.53552116, 0.38637993])},
 'Gradient Boosting': {'fit_time': array([1.20962095, 1.06314802, 1.00256324, 0.83346891, 0.85749388]),
  'score_time': array([0.01892209, 0.01110673, 0.01121306, 0.01324201, 0.011765  ]),
  'test_accuracy': array([0.80327869, 0.72131148, 0.7704918 , 0.7704918 , 0.80327869]),
  'test_precision': array([0.74179894, 0.66142857, 0.59974747, 0.63628594, 0.8297619 ]),
  'test_recall': array([0.72709552, 0.56148148, 0.66055556, 0.65264957, 0.79128205]),
  'test_r2_score': array([0.63958641, 0.61108386, 0.61108386, 0.61616651, 0.73426912]),
  'test_f1_score': array([0.73203704, 0.57548766, 0.62521008, 0.64407814, 0.80440613])},
 'Logistic Regression': {'fit_time': array([0.03856087, 0.03247309, 0.03558397, 0.03664422, 0.05600309]),
  'score_time': array([0.01866317, 0.01027703, 0.01031303, 0.01044083, 0.01519084]),
  'test_accuracy': array([0.60655738, 0.55737705, 0.63934426, 0.59016393, 0.54098361]),
  'test_precision': array([0.33070175, 0.35467492, 0.34247649, 0.5241285 , 0.3131063 ]),
  'test_recall': array([0.34736842, 0.33314815, 0.36814815, 0.39957265, 0.29606838]),
  'test_r2_score': array([0.30920729, 0.22216773, 0.50637567, 0.18804453, 0.1732817 ]),
  'test_f1_score': array([0.33403509, 0.33430353, 0.3547619 , 0.42588005, 0.28806306])}}

#perform crossvalidation to check for overfitting and how well the models perform. 
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression()
}
​
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'r2_score': make_scorer(r2_score),
           'f1_score': make_scorer(f1_score, average='macro')}
​
results = {}
​
for name, model in models.items():
    result = cross_validate(model, X1_not_G1_G2, y_train_clean, cv=5, scoring=scoring)
    results[name] = result
    print(f"{name}: Mean Accuracy: {np.mean(result['test_accuracy']):.3f}, Mean F1 Score: {np.mean(result['test_f1_score']):.3f}")
​
​
# Set up the matplotlib figure and axes
fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharey=True)
ax[0].set_title('Comparison of Accuracy')
ax[1].set_title('Comparison of Precision')
ax[2].set_title('Comparison of Recall')
ax[3].set_title('Comparison of R2 Score')
ax[4].set_title('Comparison of F1 Score')
​
index=0
# Plotting
for idx, metric in enumerate(['test_accuracy', 'test_precision', 'test_recall', 'test_r2_score', 'test_f1_score']):
    for name in results:
        ax[idx].bar(name, np.mean(results[name][metric]), label=f"{name} {metric.split('_')[1]}")
    ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
    ax[idx].set_ylabel('Score')
    index+=1
​
plt.legend()
plt.tight_layout()
plt.show()
​
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Random Forest: Mean Accuracy: 0.479, Mean F1 Score: 0.242
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Gradient Boosting: Mean Accuracy: 0.436, Mean F1 Score: 0.268
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/var/folders/7v/8544rkhj2vn_rd_7t5dmcbk80000gn/T/ipykernel_96651/2238265813.py:35: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
Logistic Regression: Mean Accuracy: 0.410, Mean F1 Score: 0.224

Fine-Tune the System
1. Pick one model and use at least one grid search to fine-tune hyperparameters

# since the best performing model in both scenarios is the XGBoost, we will use that one to perform hyperparameter 
#tuning
# Initialize the XGBoost classifier
xgb_clf = GradientBoostingClassifier()
​
# Define the parameter grid
param_grid = {
    # 'max_depth': [3, 5, 7],
    # 'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    # 'subsample': [0.8, 0.9, 1.0]
}
​
# Perform grid search
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X1_train_encoded, y_train_clean)
​
# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)
​
# Print the best score
print("Best score:", grid_search.best_score_)
​
best_model = grid_search.best_estimator_
​
Best hyperparameters: {'n_estimators': 100}
Best score: 0.6514897027875414
2. Correctly transform your testing data using your data preparation pipeline(s).

​
X_test_encoded = pipeline.transform(X_test)
y_test_clean = y_test.loc[X_test_encoded.index]
X_test_not_G1_G2 = pipe_no_g1_g2.transform(X_test)
Transforming FailuresCategorizer
Transforming AgeCategoryTransformer
Transforming TravelTimeCategoryTransformer
Transforming FamilyRelativeTransormer
Transforming FreeTimeCategoryTransformer
Transforming GooutTransformer
Transforming DailyAlcoholTransformer
Transforming BaseFeatureTransformer
Transforming EduCategoryTransformer
Transforming EduCategoryTransformer
Transforming StudyTimeCategoryTransformer
Transforming FamsizeCategoryTransformer
Transforming MJobCategoryTransformer
Transforming FJobCategoryTransformer
Transforming GuardianCategoryTransformer
Transforming DropNaTransformer
Transforming AbsencesFeatureGenerator
Transforming MinMaxScalerTransformer
Transforming OneHotEncodingTransformer
Transforming CategoricalToNumericalTransformer
Transforming FailuresCategorizer
Transforming AgeCategoryTransformer
Transforming TravelTimeCategoryTransformer
Transforming FamilyRelativeTransormer
Transforming FreeTimeCategoryTransformer
Transforming GooutTransformer
Transforming DailyAlcoholTransformer
Transforming BaseFeatureTransformer
Transforming EduCategoryTransformer
Transforming EduCategoryTransformer
Transforming StudyTimeCategoryTransformer
Transforming FamsizeCategoryTransformer
Transforming MJobCategoryTransformer
Transforming FJobCategoryTransformer
Transforming GuardianCategoryTransformer
Transforming DropNaTransformer
Transforming AbsencesFeatureGenerator
Transforming MinMaxScalerTransformer
Transforming OneHotEncodingTransformer
Transforming CategoricalToNumericalTransformer
3. Select your final model and measure its performance on the test set

#perform crossvalidation to check for overfitting and the model perofrms correctly
models = {
    "Gradient Boosting Classifier": best_model
}
​
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'r2_score': make_scorer(r2_score),
           'f1_score': make_scorer(f1_score, average='macro')}
​
results = {}
​
for name, model in models.items():
    result = cross_validate(model, X_test_encoded, y_test_clean, cv=5, scoring=scoring)
    results[name] = result
    print(f"{name}: Mean Accuracy: {np.mean(result['test_accuracy']):.3f}, Mean F1 Score: {np.mean(result['test_f1_score']):.3f}")
​
​
# Set up the matplotlib figure and axes
fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharey=True)
ax[0].set_title('Comparison of Accuracy')
ax[1].set_title('Comparison of Precision')
ax[2].set_title('Comparison of Recall')
ax[3].set_title('Comparison of R2 Score')
ax[4].set_title('Comparison of F1 Score')
​
index=0
# Plotting
for idx, metric in enumerate(['test_accuracy', 'test_precision', 'test_recall', 'test_r2_score', 'test_f1_score']):
    for name in results:
        ax[idx].bar(name, np.mean(results[name][metric]), label=f"{name} {metric.split('_')[1]}")
    ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
    ax[idx].set_ylabel('Score')
    index+=1
​
plt.legend()
plt.tight_layout()
plt.show()
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/model_selection/_split.py:725: UserWarning: The least populated class in y has only 3 members, which is less than n_splits=5.
  warnings.warn(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/var/folders/7v/8544rkhj2vn_rd_7t5dmcbk80000gn/T/ipykernel_96651/3319997124.py:33: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
Gradient Boosting Classifier: Mean Accuracy: 0.698, Mean F1 Score: 0.622


## perform crossvalidation to check for overfitting and make sure the model perofrms correctly
models = {
    "Gradient Boosting Classifier": best_model
}
​
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'r2_score': make_scorer(r2_score),
           'f1_score': make_scorer(f1_score, average='macro')}
​
results = {}
​
for name, model in models.items():
    result = cross_validate(model, X_test_not_G1_G2, y_test_clean, cv=5, scoring=scoring)
    results[name] = result
    print(f"{name}: Mean Accuracy: {np.mean(result['test_accuracy']):.3f}, Mean F1 Score: {np.mean(result['test_f1_score']):.3f}")
​
​
# Set up the matplotlib figure and axes
fig, ax = plt.subplots(5, 1, figsize=(14, 20), sharey=True)
ax[0].set_title('Comparison of Accuracy')
ax[1].set_title('Comparison of Precision')
ax[2].set_title('Comparison of Recall')
ax[3].set_title('Comparison of R2 Score')
ax[4].set_title('Comparison of F1 Score')
​
index=0
# Plotting
for idx, metric in enumerate(['test_accuracy', 'test_precision', 'test_recall', 'test_r2_score', 'test_f1_score']):
    for name in results:
        ax[idx].bar(name, np.mean(results[name][metric]), label=f"{name} {metric.split('_')[1]}")
    ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
    ax[idx].set_ylabel('Score')
    index+=1
​
plt.legend()
plt.tight_layout()
plt.show()
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/model_selection/_split.py:725: UserWarning: The least populated class in y has only 3 members, which is less than n_splits=5.
  warnings.warn(
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/amymoschos/anaconda3/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/var/folders/7v/8544rkhj2vn_rd_7t5dmcbk80000gn/T/ipykernel_96651/2716513151.py:33: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax[idx].set_xticklabels(labels=models.keys(), rotation=45)
Gradient Boosting Classifier: Mean Accuracy: 0.329, Mean F1 Score: 0.220

Present Your Solution See below
See the below additional rubric categories for the items related to presenting your solution to your executive team (in other words, presenting your work in this Jupyter notebook). Your project should include an overview and concluding section
The best model to choose for running the analysis of how to predict a student's final grades is the Gradient boosting classifier. To put it simply, the Gradient Boosting Classifier is a strong machine learning model that aims to minimize errors in data, helps capture different patterns, and works with the strengths of previous models. The accuracy is about 70% and F1 score about 62% which proves statistical significance. Based on the findings, using this model will help predict which students will need assistance in the future. The Classification task proved to work better than the regression task given the nature of the data.
