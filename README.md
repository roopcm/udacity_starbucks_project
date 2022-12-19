# Starbucks-Capstone-Challenge - Analysis of Starbucks Pilot Project
----
### Medium Blog Post
Here is the Medium blog post I have written: https://medium.com/@mitraroopc/analysis-of-starbucks-pilot-project-eddf2c2899ea


## The Dataset Used for the work
* The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.
* Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.
* As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
* There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.
* The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.

## Data Lists

### profile.json
Rewards program users (17000 users x 5 fields)

* gender: (categorical) M, F, O, or null
* age: (numeric) missing value encoded as 118
* id: (string/hash)
* became_member_on: (date) format YYYYMMDD
* income: (numeric)

### portfolio.json
Offers sent during 30-day test period (10 offers x 6 fields)

* reward: (numeric) money awarded for the amount spent
* channels: (list) web, email, mobile, social
* difficulty: (numeric) money required to be spent to receive reward
* duration: (numeric) time for offer to be open, in days
* offer_type: (string) bogo, discount, informational
* id: (string/hash)

### transcript.json
Event log (306648 events x 4 fields)

* person: (string/hash)
* event: (string) offer received, offer viewed, transaction, offer completed
* value: (dictionary) different values depending on event type
  * offer id: (string/hash) not associated with any "transaction"
  * amount: (numeric) money spent in "transaction"
  * reward: (numeric) money gained from "offer completed"
* time: (numeric) hours after start of test

<a id="files"></a>

## Files

<pre>
.
├── Starbucks_Capstone_notebook.html----------# HTML EXPORT OF JUPYTER NOTEBOOK
├── Starbucks_Capstone_notebook.ipynb---------# ANALYSIS NOTEBOOK
├── data
│   ├──transcript.json--------------------#INFORMATION ABOUT TRANSCRIPTS
    ├──portfolio.json---------------------#INFORMATION ABOUT PORTFOLIOS
│   └──profile.json-----------------------#INFORMATION ABOUT PROFILES
├── project_tests.py-----------------------# UNIT TESTS FOR PROJECT
├── picA.png-------------------------------# Figure from Udacity Template
├── picB.png-------------------------------# Figure from Udacity Template
└── reqs-----------------------------------# Requirements Folder
    ├── Starbucks_Capstone_notebook.py------#Python File genereated from Notebook
    └── requirements.txt-------------------# Libraries used
</pre>

<a id="sw_lib"></a>

## Software and Libraries

This is project uses Python 3.8.1 and the necessary libraries are mentioned in [requirements file]
in the reqs folder

The medium article on the data analysis can be found in this [link](https://medium.com/@mitraroopc/analysis-of-starbucks-pilot-project-eddf2c2899ea)
