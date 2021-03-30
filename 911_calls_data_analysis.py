# 911 Calls Capstone Project

# For this project I will be analyzing some 911 call data
# from [Kaggle](https://www.kaggle.com/mchirico/montcoalert)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic("matplotlib", "inline")

df = pd.read_csv("911.csv")

df.info()

df.head()

# What are the top 5 zipcodes for 911 calls?
df["zip"].value_counts().head()


# What are the top 5 townships (twp) for 911 calls?
df["twp"].value_counts().head()


# Take a look at the 'title' column,
# how many unique title codes are there?
df["title"].nunique()

df["lat"].round(1).value_counts()

df["lng"].round(1).value_counts()

df["latlng"] = [
    ", ".join(str(x) for x in y)
    for y in map(tuple, df[["lat", "lng"]].values.round(2))
]

df.head()

df["latlng"].value_counts()


# ## Creating new features

df["reason"] = df["title"].apply(lambda x: x.split(":")[0])
df.head()
df["reason"].value_counts()


# What is the most common Reason for a 911 call
# based off of this new column?

# Now use seaborn to create a countplot
# of 911 calls by Reason.
sns.countplot(x=df["reason"], palette="inferno")

# Now let us begin to focus on time information.
# What is the data type of the objects in the timeStamp column?
type(df["timeStamp"])
df["timeStamp"] = pd.to_datetime(df["timeStamp"])

dmap = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

dmap = {
    0: "Mon",
    1: "Tues",
    3: "Wed",
    4: "Thur",
    5: "Fri",
    6: "Sat",
    7: "Sun",
}
df["month"] = df["timeStamp"].apply(lambda x: x.month)
df["hour"] = df["timeStamp"].apply(lambda x: x.hour)
df["DOW"] = df["timeStamp"].apply(lambda x: x.dayofweek).map(dmap)

df.head()
# Now use seaborn to create a countplot
# of the Day of Week column with the hue based off
# of the Reason column.
sns.set_theme(
    style="whitegrid",
    context="notebook",
)

plt.figure(figsize=(10, 6))
sns.countplot(x="DOW", hue="reason", data=df, palette="plasma_r")

# place the legend outside the figure/plot
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.title("Reasons for 911 calls per Day of Week")
plt.tight_layout()

# Now do the same for Month
plt.figure(figsize=(10, 6))
sns.countplot(x="month", data=df, hue="reason", palette="ocean")
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.title("Reasons for 911 Calls per Month")
plt.tight_layout()


# Did you notice something strange about the Plot
# You should have noticed it was missing some Months,
# let's sene if we can maybe fill in this information by plotting
# the information in another way.
# Possibly a simple line plot that fills in the missing months,
# in order to do this, we'll need to do some work with pandas.
df["month"].value_counts()

# Interestingly there seems to be no data for months 9 thru 11.

# Now create a gropuby object called byMonth,
# where you group the DataFrame
# by the month column and use the count() method for aggregation.
# Use the head() method on this returned DataFrame.

byMonth = df.groupby("month").count()
byMonth.head()

# Now create a simple plot off of the dataframe indicating
# the count of calls per month.

plt.figure(figsize=(10, 6))
sns.lineplot(x="month", y="title", data=byMonth)


# Now see if you can use seaborn's lmplot()
# to create a linear fit on the
# number of calls per month.
# Keep in mind you may need to reset the index to a column.

plt.figure(figsize=(10, 60))
sns.lmplot(x="month", y="title", data=byMonth.reset_index())


# Create a new column called 'Date' that contains
# the date from the timeStamp column.
# You'll need to use apply along with the .date() method.
df["date"] = df["timeStamp"].apply(lambda x: x.date())

df.head()


# Now groupby this Date column with the count()
# aggregate and create a plot of counts of 911 calls

byDate = df.groupby("date").count()
plt.figure(figsize=(10, 6))
sns.lineplot(x="date", y="title", data=byDate)
plt.xlabel("date")
plt.ylabel("911 call count")
plt.title("call volume by date")


# Now recreate this plot but create 3 separate plots
# with each plot representing a Reason for the 911 cal

date_ems = df[df["reason"] == "EMS"].groupby("date").count()
plt.figure(figsize=(10, 6))
sns.lineplot(x="date", y="title", data=date_ems)
plt.xlabel("date")
plt.ylabel("911 call count")
plt.title("ems")


date_traffic = df[df["reason"] == "Traffic"].groupby("date").count()
plt.figure(figsize=(10, 6))
sns.lineplot(x="date", y="title", data=date_traffic)
plt.xlabel("date")
plt.ylabel("911 call count")
plt.title("traffic")


date_fire = df[df["reason"] == "Fire"].groupby("date").count()
plt.figure(figsize=(10, 6))
sns.lineplot(x="date", y="title", data=date_fire)
plt.xlabel("date")
plt.ylabel("911 call count")
plt.title("fire")


# Now let's move on to creating  heatmaps with seaborn
# and our data.
# We'll first need to restructure the dataframe so that
# the columns become the Hours and the Index
# becomes the Day of the Week.

dayHour = df.groupby(["DOW", "hour"]).count()["reason"].unstack()
dayHour

# Now create a HeatMap using this new DataFrame.


plt.figure(figsize=(10, 6))
sns.heatmap(dayHour)
plt.title("call frequency by day and hour")
plt.xlabel("hour")
plt.ylabel("day of the week")


# Now create a clustermap using this DataFrame.

plt.figure(figsize=(10, 6))
sns.clustermap(dayHour)
plt.title("call frequency by day and hour")


# Now repeat these same plots and operations,
# for a DataFrame that shows the Month as the column.

monthHour = df.groupby(["month", "hour"]).count()["reason"].unstack()
monthHour

plt.figure(figsize=(10, 6))
sns.heatmap(monthHour)
plt.title("call frequency by month and hour")
plt.xlabel("hour")
plt.ylabel("month")


plt.figure(figsize=(10, 6))
sns.clustermap(dayHour)
plt.title("call frequency by month and hour")


plt.figure(figsize=(18, 6))
sns.countplot(x="twp", hue="reason", data=df, palette="plasma_r")

# place the legend outside the figure/plot
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.title("Reasons for 911 calls per Day of Week")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("911Call_reasons_township.png")
