# rmuraglia_jobtest_explore.py

"""
Ryan Muraglia's code submission for [job application] coding assignment
Testing period: Oct 13 2016 - Oct 18 2016
"""

# Python version 2.7.10
import numpy as np # version 1.11.1
import pandas as pd # version 0.18.1
import matplotlib.pyplot as plt # version 1.3.1

data_dir = '~/Desktop/jobtest/'

"""
Part 0: Explore raw data
Note: This script does not need to be run. I simply like to work through the raw data and take notes in this manner to get more acquainted with the format and variable types, and made it available to more accurately represent my workflow.
"""

# load raw data
all_raw = pd.read_csv(data_dir + 'Data for Cleaning & Modeling.csv')
metadata = pd.read_csv(data_dir +  'Metadata.csv', index_col=0)

# see dimensions
print all_raw.shape

# inspect each variable
for col in all_raw.columns :
    print 'Variable "%s" represents "%s"' % (col, metadata.loc[col][0])
    uniques = all_raw[col].unique()
    print 'There are %i unique values for this variable' % (len(uniques))
    print 'Here are some examples: '
    print uniques[0:5]
    print 'Currently these are encoded as "%s" variables' % (uniques[0].__class__)
    print 'There are %i missing values for this variable' % (len(np.where(all_raw[col].isnull())[0]))
    raw_input('Press enter to proceed to the next variable... \n')

# for some intuitively interesting/promising variables, make some exploratory plots
# subset 5000 random draws for plotting
mini_raw = all_raw.sample(n=5000)
X1_plot = mini_raw['X1'].str.strip('%').astype('float')
mini_raw['X1'] = X1_plot

# interest rate vs loan amount
# unclear pattern
X4_plot = mini_raw['X4'].str.replace('[\$,]', '').astype('float')
X5_plot = mini_raw['X5'].str.replace('[\$,]', '').astype('float')
X6_plot = mini_raw['X6'].str.replace('[\$,]', '').astype('float')

fig1 = plt.figure(1)
ax1 = plt.subplot(131)
ax1.plot(X4_plot, X1_plot, marker='o', linestyle='none')
ax1.set_ylabel('Interest Rate')
ax1.set_title('Requested')

ax2 = plt.subplot(132)
ax2.plot(X5_plot, X1_plot, marker='o', linestyle='none')
ax2.set_xlabel('Loan Amount')
ax2.set_title('Funded')

ax3 = plt.subplot(133)
ax3.plot(X6_plot, X1_plot, marker='o', linestyle='none')
ax3.set_title('Investor Portion')

plt.show()

# interest rate vs number of payments
# clear trend
mini_raw[['X1', 'X7']].boxplot(by='X7'); plt.show()

# interest rate vs loan grade
# clear trend
# note: many missing values in training set, but none in test set
mini_raw[['X1', 'X9']].boxplot(by='X9'); plt.show()

# vs number years employed
# unclear
mini_raw[['X1', 'X11']].boxplot(by='X11'); plt.show()

# vs home status
# unclear
mini_raw[['X1', 'X12']].boxplot(by='X12'); plt.show()

# vs income
# income is very skewed
mini_raw.plot(x='X13', y='X1', kind='scatter'); plt.show()

# vs income verification
# unverified income seems to have lowest interest rates - unintuitive.
mini_raw[['X1', 'X14']].boxplot(by='X14'); plt.show()

# vs issue date
# possibly some seasonal trends? hard to tell
mini_raw[['X1', 'X15']].boxplot(by='X15'); plt.show()

# vs loan category
# can be predictive
mini_raw[['X1', 'X17']].boxplot(by='X17'); plt.show()

# vs state
# has potential
mini_raw[['X1', 'X20']].boxplot(by='X20'); plt.show()

# vs total monthly debt payments/monthly income
# unclear
mini_raw.plot(x='X21', y='X1', kind='scatter'); plt.show()

# vs negative credit marks: delinquency, inquiries, derogatory records
# unclear trends
fig = plt.figure()
ax1 = plt.subplot(131)
ax1.plot(mini_raw['X22'], X1_plot, marker='o', linestyle='none')
ax1.set_ylabel('Interest Rate')
ax1.set_xlabel('# Deliquencies')
ax2 = plt.subplot(132)
ax2.plot(mini_raw['X24'], X1_plot, marker='o', linestyle='none')
ax2.set_xlabel('# Inquiries')
ax3 = plt.subplot(133)
ax3.plot(mini_raw['X28'], X1_plot, marker='o', linestyle='none')
ax3.set_xlabel('# Derogatory Records')
plt.show()

# vs time since last bad mark
# unclear
# note: many NaNs (possibly for those with no previous bad marks)
fig = plt.figure()
ax1 = plt.subplot(121)
ax1.plot(mini_raw['X25'], X1_plot, marker='o', linestyle='none')
ax1.set_ylabel('Interest Rate')
ax1.set_xlabel('Months since delinquency')
ax2 = plt.subplot(122)
ax2.plot(mini_raw['X26'], X1_plot, marker='o', linestyle='none')
ax2.set_xlabel('Months since public record')
plt.show()

# vs revolving credit balance and utilization rate
# revolving balance is skewed (like income)
# utilization rate may be predictive, but there are missing values in test set, so will require imputation method
X30_plot = mini_raw['X30'].str.strip('%').astype('float')
fig = plt.figure()
ax1 = plt.subplot(121)
ax1.plot(mini_raw['X29'], X1_plot, marker='o', linestyle='none')
ax1.set_ylabel('Interest Rate')
ax1.set_xlabel('Revolving Balance')
ax2 = plt.subplot(122)
ax2.plot(X30_plot, X1_plot, marker='o', linestyle='none')
ax2.set_xlabel('Utilization Rate')
plt.show()

# vs number of credit lines
# unclear trend
fig = plt.figure()
ax1 = plt.subplot(121)
ax1.plot(mini_raw['X27'], X1_plot, marker='o', linestyle='none')
ax1.set_ylabel('Interest Rate')
ax1.set_xlabel('# open credit lines')
ax2 = plt.subplot(122)
ax2.plot(mini_raw['X31'], X1_plot, marker='o', linestyle='none')
ax2.set_xlabel('Total # credit lines')
plt.show()

# vs loan status
# unclear
mini_raw[['X1', 'X32']].boxplot(by='X32'); plt.show()

# alternate views for skewed income related features
fig = plt.figure()
ax1 = plt.subplot(231)
ax1.plot(mini_raw['X13'], mini_raw['X1'], marker='o', linestyle='none')
ax1.set_title('Income')
ax2 = plt.subplot(232)
ax2.plot(mini_raw['X13'], mini_raw['X1'], marker='o', linestyle='none')
ax2.set_xlim([0, 100000])
ax2.set_title('Income Zoom')
ax3 = plt.subplot(233)
ax3.plot(np.log(mini_raw['X13']), X1_plot, marker='o', linestyle='none')
ax3.set_title('log(Income)')
ax4 = plt.subplot(234)
ax4.plot(mini_raw['X29'], mini_raw['X1'], marker='o', linestyle='none')
ax4.set_title('Rev Bal')
ax5 = plt.subplot(235)
ax5.plot(mini_raw['X29'], mini_raw['X1'], marker='o', linestyle='none')
ax5.set_xlim([0, 50000])
ax5.set_title('Rev Bal Zoom')
ax6 = plt.subplot(236)
ax6.plot(np.log(mini_raw['X29']), X1_plot, marker='o', linestyle='none')
ax6.set_title('log(Rev Bal)')
plt.show()

# check for collinearity between X4, X5 and X6 variables
# appears collinear. Note that X4 >= X5, X5 >= X6
fig = plt.figure()
ax1 = plt.subplot(131)
ax1.plot(X4_plot, X5_plot, marker='o', linestyle='none')
ax1.set_xlabel('X4'); ax1.set_ylabel('X5')
ax2 = plt.subplot(132)
ax2.plot(X4_plot, X6_plot, marker='o', linestyle='none')
ax2.set_xlabel('X4'); ax2.set_ylabel('X6')
ax3 = plt.subplot(133)
ax3.plot(X6_plot, X5_plot, marker='o', linestyle='none')
ax3.set_xlabel('X6'); ax3.set_ylabel('X5')
plt.show()