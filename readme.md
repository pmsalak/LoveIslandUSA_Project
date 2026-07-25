# Final Project: Exploring Love Island USA Data


# Background Information & Data Preparation

For my final project, I found a data set containing information on all
Love Island USA contestants, aka ‘islanders’, from seasons 1-6. Love
Island is a reality TV show where single people live in a villa in Fiji
all summer while searching for love. While one group enters the villa on
day 1, several more people are introduced into the mix as ‘bombshells’
throughout the season. In order to remain in the villa, islanders must
be coupled up with a partner or they will be dumped. Each season ends
with America voting for their favorite remainning couple as the winners.

Loading the packages I am going to use:

``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

My data is a csv file that I downloaded from GitHub. Importing my data:

``` python
love_island_df = pd.read_csv('C:/Users/Parri Salak/OneDrive/Documents/MSBA/Python/LoveIslandUSA.csv')
```

Cleaning column names to be in snake case:

``` python
love_island_df.columns
love_island_df.columns = love_island_df.columns.str.strip().str.replace(' ', '_')
love_island_df.columns = love_island_df.columns.str.lower()
love_island_df.columns
```

    Index(['season', 'sex', 'first_name', 'last_name', 'age', 'occupation',
           'result', 'first_group_in_villa',
           'have_they_appeared_on_a_previous_season_of_li_usa?',
           'day_entered_villa', 'day_left_villa', 'days_in_villa', 'location',
           'us_region', 'birthday', 'zodiac', 'casa_amor_contestant',
           'casa_decision'],
          dtype='str')

# Data Exploration

Looking at the first 5 rows of data:

``` python
love_island_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | season | sex | first_name | last_name | age | occupation | result | first_group_in_villa | have_they_appeared_on_a_previous_season_of_li_usa? | day_entered_villa | day_left_villa | days_in_villa | location | us_region | birthday | zodiac | casa_amor_contestant | casa_decision |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 1 | Female | Elizabeth | Weber | 24 | Advertising executive | Winner | Yes | No | 1 | 27 | 26 | New York, New York | Northeast | 8-Sep-94 | Virgo | No Casa this season | No Casa this season |
| 1 | 1 | Male | Zac | Mirabelli | 22 | Grocery store cashier | Winner | Yes | No | 1 | 27 | 26 | Chicago, Illinois | Midwest | 18-Sep-96 | Virgo | No Casa this season | No Casa this season |
| 2 | 1 | Female | Alexandra | Stewart | 25 | Publicist | Runner-Up | Yes | No | 1 | 27 | 26 | Los Angeles, California | West | 4-Jun-94 | Gemini | No Casa this season | No Casa this season |
| 3 | 1 | Male | Dylan | Curry | 25 | Personal trainer | Runner-Up | No | No | 4 | 27 | 23 | San Diego, California | West | 3-Feb-94 | Aquarius | No Casa this season | No Casa this season |
| 4 | 1 | Female | Caro | Viehweg | 21 | Student | 3rd Place | Yes | No | 1 | 27 | 26 | Los Angeles, California | West | 26-Dec-97 | Capricorn | No Casa this season | No Casa this season |

</div>

``` python
love_island_df.shape
```

    (190, 18)

The data contains 190 rows, and 18 columns.

``` python
love_island_df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 190 entries, 0 to 189
    Data columns (total 18 columns):
     #   Column                                              Non-Null Count  Dtype
    ---  ------                                              --------------  -----
     0   season                                              190 non-null    int64
     1   sex                                                 190 non-null    str  
     2   first_name                                          190 non-null    str  
     3   last_name                                           190 non-null    str  
     4   age                                                 190 non-null    int64
     5   occupation                                          185 non-null    str  
     6   result                                              190 non-null    str  
     7   first_group_in_villa                                190 non-null    str  
     8   have_they_appeared_on_a_previous_season_of_li_usa?  190 non-null    str  
     9   day_entered_villa                                   190 non-null    int64
     10  day_left_villa                                      190 non-null    int64
     11  days_in_villa                                       190 non-null    int64
     12  location                                            190 non-null    str  
     13  us_region                                           190 non-null    str  
     14  birthday                                            164 non-null    str  
     15  zodiac                                              164 non-null    str  
     16  casa_amor_contestant                                190 non-null    str  
     17  casa_decision                                       82 non-null     str  
    dtypes: int64(5), str(13)
    memory usage: 26.8 KB

``` python
love_island_df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|       | season     | age        | day_entered_villa | day_left_villa | days_in_villa |
|-------|------------|------------|-------------------|----------------|---------------|
| count | 190.000000 | 190.000000 | 190.000000        | 190.000000     | 190.000000    |
| mean  | 3.621053   | 24.789474  | 9.868421          | 23.942105      | 14.073684     |
| std   | 1.659852   | 2.381119   | 8.390506          | 9.013324       | 11.598433     |
| min   | 1.000000   | 21.000000  | 1.000000          | 4.000000       | 2.000000      |
| 25%   | 2.000000   | 23.000000  | 1.000000          | 19.000000      | 4.000000      |
| 50%   | 4.000000   | 24.000000  | 10.000000         | 24.000000      | 9.000000      |
| 75%   | 5.000000   | 27.000000  | 16.750000         | 31.000000      | 23.750000     |
| max   | 6.000000   | 32.000000  | 30.000000         | 40.000000      | 39.000000     |

</div>

## Age

One variable I would like to further explore is age. Most islanders that
appear on the show are in their 20s, but what is their average age? Does
this change across different groups, such as season, gender, or result?

``` python
love_island_df['age'].max()
```

    np.int64(32)

``` python
oldest_islander = love_island_df[love_island_df['age'] == love_island_df['age'].max()]
oldest_islander['first_name'] + ' ' + oldest_islander['last_name']
```

    121    Felipe Gomes
    dtype: str

The oldest islander was 32 years old at the time of their appearance on
Love Island. This islander was Felipe Gomes.

``` python
love_island_df['age'].mean()
sns.histplot(data=love_island_df, x='age', bins=10)
plt.show()
```

![](readme_files/figure-commonmark/cell-11-output-1.png)

The average age of contestants on Love Island is ~25 years old. However,
the data also appear to be skewed right.

``` python
love_island_df.groupby('season')['age'].mean()
```

    season
    1    24.680000
    2    24.677419
    3    25.411765
    4    24.676471
    5    24.151515
    6    25.090909
    Name: age, dtype: float64

Average age seems to have remained consistant across seasons, with
season 3 having the highest average age and season 5 having the lowest.

``` python
love_island_df.groupby('sex')['age'].mean()
```

    sex
    Female    24.170213
    Male      25.395833
    Name: age, dtype: float64

Female islanders have a slightly lower average age then male islanders.

``` python
love_island_df.groupby('result')['age'].mean()
```

    result
    3rd Place    24.916667
    4th Place    24.900000
    Dumped       24.906250
    Removed      24.000000
    Runner-Up    23.500000
    Walked       24.428571
    Winner       25.166667
    Name: age, dtype: float64

The average age of winners does not differ from the overall average age
of contestants.

## Location

Another variable I would like to explore is location. This is the city
that each islander is currently living in before appearing on Love
Island.

``` python
love_island_df['location'].nunique()
```

    127

There are 127 different locations where islanders come from.

``` python
love_island_df['location'].value_counts().sort_values(ascending=False)
```

    location
    Los Angeles, California          16
    Miami, Florida                   10
    Houston, Texas                   10
    New York, New York                5
    Dallas, Texas                     5
                                     ..
    Athens, Georgia                   1
    Medellín, Colombia                1
    Hagerstown, Indiana               1
    Santa Monica, California          1
    Winston-Salem, North Carolina     1
    Name: count, Length: 127, dtype: int64

The most common city for islanders to be living in is Los Angeles,
California with 16 contestants. This is most likely because producers
host in-person recruiting for the show in LA.

## Days in Villa

As more contestants enter the villa, islanders are forced to leave. I
want to look at the minimum, maximum, and average amount of time spent
in the villa.

``` python
love_island_df['days_in_villa'].describe()
```

    count    190.000000
    mean      14.073684
    std       11.598433
    min        2.000000
    25%        4.000000
    50%        9.000000
    75%       23.750000
    max       39.000000
    Name: days_in_villa, dtype: float64

The longest time spent in the villa was 39 days, and the shortest was
only 2 days. The average time in the villa is around 14 days, or two
weeks.

## Predicting Winners

(first_group_in_villa as predictor).. one hot encoding?

## Conclusion
