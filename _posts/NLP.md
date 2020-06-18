{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NLP With Hotel Review Part 2\n",
    "##### By: David Penny"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Goal: develop several machine learning models to correctly label the sentiment behind hotel reviews.\n",
    "\n",
    "Process:\n",
    "\n",
    "- Exploratory Data Analysis (EDA)\n",
    "- Data augmentation\n",
    "- Modelling\n",
    "- Iteration over model improvements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Filter warnings\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part 1: EDA - Basics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#First up, lets read in the data.\n",
    "df_train = pd.read_csv(\"data/train_dataframe.csv\")\n",
    "df_test = pd.read_csv(\"data/test_dataframe.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's see all the columns just for fun\n",
    "# for col_name in df_test.columns: \n",
    "#     print(col_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Additional_Number_of_Scoring</th>\n",
       "      <th>Average_Score</th>\n",
       "      <th>Review_Total_Negative_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews</th>\n",
       "      <th>Review_Total_Positive_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>lat</th>\n",
       "      <th>lng</th>\n",
       "      <th>Review_Month</th>\n",
       "      <th>...</th>\n",
       "      <th>p_working</th>\n",
       "      <th>p_world</th>\n",
       "      <th>p_worth</th>\n",
       "      <th>p_wouldn</th>\n",
       "      <th>p_year</th>\n",
       "      <th>p_years</th>\n",
       "      <th>p_yes</th>\n",
       "      <th>p_young</th>\n",
       "      <th>p_yummy</th>\n",
       "      <th>Reviewer_Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>220.0</td>\n",
       "      <td>9.1</td>\n",
       "      <td>20.0</td>\n",
       "      <td>902.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>275.0</td>\n",
       "      <td>51.494308</td>\n",
       "      <td>-0.175558</td>\n",
       "      <td>11.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1190.0</td>\n",
       "      <td>7.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5180.0</td>\n",
       "      <td>23.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>481.0</td>\n",
       "      <td>51.514879</td>\n",
       "      <td>-0.160650</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.425849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>299.0</td>\n",
       "      <td>8.3</td>\n",
       "      <td>81.0</td>\n",
       "      <td>1361.0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>672.0</td>\n",
       "      <td>51.521009</td>\n",
       "      <td>-0.123097</td>\n",
       "      <td>10.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>87.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>355.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>412.0</td>\n",
       "      <td>51.499749</td>\n",
       "      <td>-0.161524</td>\n",
       "      <td>6.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>317.0</td>\n",
       "      <td>7.6</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1458.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>499.0</td>\n",
       "      <td>51.516114</td>\n",
       "      <td>-0.174952</td>\n",
       "      <td>3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2587 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Additional_Number_of_Scoring  Average_Score  \\\n",
       "0                         220.0            9.1   \n",
       "1                        1190.0            7.5   \n",
       "2                         299.0            8.3   \n",
       "3                          87.0            9.0   \n",
       "4                         317.0            7.6   \n",
       "\n",
       "   Review_Total_Negative_Word_Counts  Total_Number_of_Reviews  \\\n",
       "0                               20.0                    902.0   \n",
       "1                                5.0                   5180.0   \n",
       "2                               81.0                   1361.0   \n",
       "3                               17.0                    355.0   \n",
       "4                               14.0                   1458.0   \n",
       "\n",
       "   Review_Total_Positive_Word_Counts  \\\n",
       "0                               21.0   \n",
       "1                               23.0   \n",
       "2                               27.0   \n",
       "3                               13.0   \n",
       "4                                0.0   \n",
       "\n",
       "   Total_Number_of_Reviews_Reviewer_Has_Given  days_since_review        lat  \\\n",
       "0                                         1.0              275.0  51.494308   \n",
       "1                                         6.0              481.0  51.514879   \n",
       "2                                         4.0              672.0  51.521009   \n",
       "3                                         7.0              412.0  51.499749   \n",
       "4                                         1.0              499.0  51.516114   \n",
       "\n",
       "        lng  Review_Month  ...  p_working  p_world  p_worth  p_wouldn  p_year  \\\n",
       "0 -0.175558          11.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "1 -0.160650           4.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "2 -0.123097          10.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "3 -0.161524           6.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "4 -0.174952           3.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "\n",
       "    p_years  p_yes  p_young  p_yummy  Reviewer_Score  \n",
       "0  0.000000    0.0      0.0      0.0             1.0  \n",
       "1  0.425849    0.0      0.0      0.0             1.0  \n",
       "2  0.000000    0.0      0.0      0.0             0.0  \n",
       "3  0.000000    0.0      0.0      0.0             1.0  \n",
       "4  0.000000    0.0      0.0      0.0             0.0  \n",
       "\n",
       "[5 rows x 2587 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Additional_Number_of_Scoring</th>\n",
       "      <th>Average_Score</th>\n",
       "      <th>Review_Total_Negative_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews</th>\n",
       "      <th>Review_Total_Positive_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>lat</th>\n",
       "      <th>lng</th>\n",
       "      <th>Review_Month</th>\n",
       "      <th>...</th>\n",
       "      <th>p_working</th>\n",
       "      <th>p_world</th>\n",
       "      <th>p_worth</th>\n",
       "      <th>p_wouldn</th>\n",
       "      <th>p_year</th>\n",
       "      <th>p_years</th>\n",
       "      <th>p_yes</th>\n",
       "      <th>p_young</th>\n",
       "      <th>p_yummy</th>\n",
       "      <th>Reviewer_Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2288.0</td>\n",
       "      <td>8.1</td>\n",
       "      <td>24.0</td>\n",
       "      <td>9568.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>527.0</td>\n",
       "      <td>51.511099</td>\n",
       "      <td>-0.120867</td>\n",
       "      <td>2.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>61.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>263.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>527.0</td>\n",
       "      <td>51.522636</td>\n",
       "      <td>-0.160287</td>\n",
       "      <td>2.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>974.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>3040.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>485.0</td>\n",
       "      <td>51.500732</td>\n",
       "      <td>-0.016550</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>838.0</td>\n",
       "      <td>8.4</td>\n",
       "      <td>23.0</td>\n",
       "      <td>3274.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>415.0</td>\n",
       "      <td>51.495666</td>\n",
       "      <td>-0.145279</td>\n",
       "      <td>6.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>879.0</td>\n",
       "      <td>8.8</td>\n",
       "      <td>48.0</td>\n",
       "      <td>2768.0</td>\n",
       "      <td>51.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>385.0</td>\n",
       "      <td>51.508354</td>\n",
       "      <td>0.019886</td>\n",
       "      <td>7.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2587 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Additional_Number_of_Scoring  Average_Score  \\\n",
       "0                        2288.0            8.1   \n",
       "1                          61.0            9.0   \n",
       "2                         974.0            9.0   \n",
       "3                         838.0            8.4   \n",
       "4                         879.0            8.8   \n",
       "\n",
       "   Review_Total_Negative_Word_Counts  Total_Number_of_Reviews  \\\n",
       "0                               24.0                   9568.0   \n",
       "1                                0.0                    263.0   \n",
       "2                               20.0                   3040.0   \n",
       "3                               23.0                   3274.0   \n",
       "4                               48.0                   2768.0   \n",
       "\n",
       "   Review_Total_Positive_Word_Counts  \\\n",
       "0                               16.0   \n",
       "1                               21.0   \n",
       "2                               20.0   \n",
       "3                               29.0   \n",
       "4                               51.0   \n",
       "\n",
       "   Total_Number_of_Reviews_Reviewer_Has_Given  days_since_review        lat  \\\n",
       "0                                         1.0              527.0  51.511099   \n",
       "1                                         1.0              527.0  51.522636   \n",
       "2                                        12.0              485.0  51.500732   \n",
       "3                                         5.0              415.0  51.495666   \n",
       "4                                         1.0              385.0  51.508354   \n",
       "\n",
       "        lng  Review_Month  ...  p_working  p_world  p_worth  p_wouldn  p_year  \\\n",
       "0 -0.120867           2.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "1 -0.160287           2.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "2 -0.016550           4.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "3 -0.145279           6.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "4  0.019886           7.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "\n",
       "   p_years  p_yes  p_young  p_yummy  Reviewer_Score  \n",
       "0      0.0    0.0      0.0      0.0             0.0  \n",
       "1      0.0    0.0      0.0      0.0             1.0  \n",
       "2      0.0    0.0      0.0      0.0             0.0  \n",
       "3      0.0    0.0      0.0      0.0             1.0  \n",
       "4      0.0    0.0      0.0      0.0             1.0  \n",
       "\n",
       "[5 rows x 2587 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Additional_Number_of_Scoring         0\n",
      "Average_Score                        0\n",
      "Review_Total_Negative_Word_Counts    0\n",
      "Total_Number_of_Reviews              0\n",
      "Review_Total_Positive_Word_Counts    0\n",
      "                                    ..\n",
      "p_years                              0\n",
      "p_yes                                0\n",
      "p_young                              0\n",
      "p_yummy                              0\n",
      "Reviewer_Score                       0\n",
      "Length: 2587, dtype: int64\n",
      "Additional_Number_of_Scoring         0\n",
      "Average_Score                        0\n",
      "Review_Total_Negative_Word_Counts    0\n",
      "Total_Number_of_Reviews              0\n",
      "Review_Total_Positive_Word_Counts    0\n",
      "                                    ..\n",
      "p_years                              0\n",
      "p_yes                                0\n",
      "p_young                              0\n",
      "p_yummy                              0\n",
      "Reviewer_Score                       0\n",
      "Length: 2587, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#Check step! Lets see how clean our data is\n",
    "print(df_test.isna().sum())\n",
    "print(df_train.isna().sum())\n",
    "#Looks good!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(13651, 2587)\n",
      "(3413, 2587)\n"
     ]
    }
   ],
   "source": [
    "print(df_train.shape)\n",
    "print(df_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.2500183136766537"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's see what the split is for our train-test split\n",
    "df_test.shape[0]/df_train.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 1.1 Fit a logisitic regression model to this data with the solver set to lbfgs. What is the accuracy score on the test set?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting up our variables\n",
    "y_train = df_train['Reviewer_Score']\n",
    "X_train = df_train.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "y_test = df_test['Reviewer_Score']\n",
    "X_test = df_test.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "# We'll need these later\n",
    "y_train2 = y_train\n",
    "X_train2 = X_train\n",
    "\n",
    "y_test2 = y_test\n",
    "X_test2 = X_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The train accuracy is: 0.721\n",
      "The test accuracy is: 0.719\n"
     ]
    }
   ],
   "source": [
    "# CAUTION: LONG RUN TIME (~5mins)\n",
    "# Instantiate and train the classifier\n",
    "logistic_regression_model = LogisticRegression(solver='lbfgs', max_iter=100)\n",
    "logistic_regression_model.fit(X_train, y_train)\n",
    "# Evaluate it\n",
    "\n",
    "print(f'The train accuracy is: {logistic_regression_model.score(X_train,y_train):0.3f}')\n",
    "print(f'The test accuracy is: {logistic_regression_model.score(X_test,y_test):0.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The train accuracy is: 0.811\n",
    "- The test accuracy is: 0.785"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 1.2 What are the 20 words most predictive of a good review (from the positive review column)? What are the 20 words most predictive with a bad review (from the negative review column)? Use the regression coefficients to answer this question"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.37777223e-04,  1.14138748e-01, -4.66590998e-02, ...,\n",
       "         1.55338632e-05, -1.03301628e-04,  5.15123237e-05]])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "coefficients = logistic_regression_model.coef_\n",
    "coefficients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "indices = coefficients.argsort()[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<2586x2653 sparse matrix of type '<class 'numpy.int64'>'\n",
       "\twith 3648 stored elements in Compressed Sparse Row format>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 1. Instantiate\n",
    "bagofwords = CountVectorizer()\n",
    "\n",
    "# 2. Fit\n",
    "bagofwords.fit(X_train)\n",
    "\n",
    "# 3. Transform\n",
    "X_train = bagofwords.transform(X_train)\n",
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['angel', '82', 'p_receptionist', 'p_milk', 'n_signs',\n",
       "       'p_housekeeping', 'cannizaro', 'n_stopped', 'n_car', 'n_simple',\n",
       "       'n_tasted', 'n_calls', 'p_perfectly', 'n_peeling',\n",
       "       'additional_number_of_scoring', 'n_used', 'n_realise', 'n_use',\n",
       "       'n_drink', 'george'], dtype='<U42')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The words with the lowest coefficients \n",
    "# most predictive of a 0 (negative review)\n",
    "np.array(bagofwords.get_feature_names())[indices[:20]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['p_public', 'p_cake', 'p_sleep', 'p_slept', 'p_special', 'p_think',\n",
       "       'p_happy', 'p_king', 'p_generally', 'p_italian', 'p_bed', 'p_hyde',\n",
       "       'hotel_name_portobello', 'hotel_name_conrad', 'p_movies',\n",
       "       'p_forward', 'p_terrace', 'n_parking', 'academy', '55'],\n",
       "      dtype='<U42')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The words with the highest coefficients\n",
    "# most predictive of a 1 (positive review)\n",
    "np.array(bagofwords.get_feature_names())[indices[-20:]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Thoughts:\n",
    "- Its funny, I'm getting some funny words ending up in the coeffecients. What happens if I use only the words from the postive and negative review columns?\n",
    "\n",
    "(Omit below, did not use)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_neg = df_train.filter(regex='n_', axis=1);\n",
    "# df_pos = df_train.filter(regex='p_', axis=1);\n",
    "# df_words_train = pd.concat([df_neg, df_pos], axis=1, sort=False)\n",
    "\n",
    "# #Lets store the phrases here:\n",
    "# X_train = df_words_train\n",
    "# #Lets store the sentiment here:\n",
    "# y_train = df_train['Reviewer_Score']\n",
    "\n",
    "# print(X_train.shape)\n",
    "# print(y_train.shape)\n",
    "\n",
    "# # 1. Instantiate\n",
    "# bagofwords = CountVectorizer()\n",
    "\n",
    "# # 2. Fit\n",
    "# bagofwords.fit(X_train)\n",
    "\n",
    "# # 3. Transform\n",
    "# X_train = bagofwords.transform(X_train)\n",
    "# X_train\n",
    "\n",
    "# print(X_train.shape)\n",
    "# print(y_train.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 1.3 Reduce the dimensionality of the dataset using PCA, what is the relationship between the number of dimensions and run-time for a logistic regression?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Need to reset our variables...\n",
    "y_train = df_train['Reviewer_Score']\n",
    "X_train = df_train.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "y_test = df_test['Reviewer_Score']\n",
    "X_test = df_test.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "# We'll need these later\n",
    "y_train2 = y_train\n",
    "X_train2 = X_train\n",
    "\n",
    "y_test2 = y_test\n",
    "X_test2 = X_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train = scaler.transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build and fit a PCA model to the data\n",
    "\n",
    "# 1. Instantiate\n",
    "my_pca = PCA(n_components=10)\n",
    "\n",
    "# 2. Fit (mathematical calculations are made at this step) \n",
    "my_pca.fit(X_train)\n",
    "\n",
    "# 3. Transform\n",
    "X_PCA = my_pca.transform(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13651, 10)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# looking at transformed data\n",
    "X_PCA.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 3.93439903e-02, -1.04542189e-01,  3.36051025e-01, ...,\n",
       "        -3.93758381e-03,  2.69421820e-03, -1.57205080e-03],\n",
       "       [-4.02368699e-02,  1.15938789e-01,  7.06388496e-02, ...,\n",
       "        -1.52626273e-03,  9.88123067e-03,  5.67662909e-04],\n",
       "       [-9.04637953e-02, -1.81646683e-01, -2.69949061e-02, ...,\n",
       "         4.54168960e-03,  3.10174487e-03, -6.52892287e-03],\n",
       "       ...,\n",
       "       [ 7.99274373e-02, -1.37982455e-01, -4.12776930e-02, ...,\n",
       "         7.18470312e-03, -4.55682921e-03,  1.07996667e-02],\n",
       "       [ 8.19159640e-02,  1.30361440e-02, -4.45165871e-02, ...,\n",
       "         2.59532892e-03, -1.27119963e-02, -1.06133224e-03],\n",
       "       [-4.59937198e-02,  2.27248183e-02,  2.74310402e-03, ...,\n",
       "        -9.40469951e-04,  2.38976021e-04,  1.13144330e-02]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# the principal components\n",
    "my_pca.components_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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eF0mAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAAI5EZAAAAAAAAAAAA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AKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgAAAAAAAAAAAKREZgBwR9M0xel0immalp4CAAAAAAAAAO8iMgOAOxnHMfq+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7vUEzCJ+tv7DXL/jhr2e4xpRmTe66+bNm/GHf/iHJ77++PjSaY4XHMc597smjx49ij/+4z8eV5nnMq57oN91Oa67F8/6TASbCAkAACAASURBVOx3n33nO985ETCLGN+emNbze55+TrB4ZnWcLwAAAADAKITMAACmaJgAzzhCKeMKLA1T/9WrV+Pb3/523Lt3r29nsbzQ0yjOEu46izRN49atWyeO3759OyqVypkCWMNcz3GH6PICNc8//3xcu3at59jxdRxXKOo04zrnWq3WNwT1B3/wB3Hnzp1xlXsu47gHjl+Xq1evnrgnzhv8yguRHre8vDyWPSF4w0UwqZ/HAAAAAADTJmQGADBFwwZ4uqGU73//+/EXf/EXsbGxMfRnnBbeOU+w43j9pVIpnjx5Er/7u78bzz33XO4fz8/bRWha3bXu378fly71/oq8srISzz77bEScPYB1WshoWiG6Wq0Wjx8/7jnWbx3HEYo6Td65jXrOlUolXnvttb6vvfLKKzMTYBpHJ62j1+Xdd9+Nt956a6i9OOw9P0y3tOXl5bh///6590Re8EbwjEUyzS6cAAAAAACTJmQGAFxYRYUZhg3wbG9vxwsvvBAvvvhiVKvVuHPnzol6+53DoMDSODqqHA3AXbp0KQ4ODib+x/OzhrtGucbNZjNeeOGF+OCDD3qOP3nypCeEddYA1qCQ0bRCdKOs46THC66srES73e451m63Y2VlZeT3unnzZiwvL584Pomg3iSMsk+PXpdh9uIo93y//fHrv/7rPf/+3e9+N65fv37u8+0XvLlz546OTyyUaQWIAQAAAACmIcmyrOgahra2tpbt7u4WXQYAsACazWbU6/UolUrR6XSi0WhMpFvTWaVpGtVq9UQI59q1a/H48eNoNBoREX3Pod/3lsvluHv3bjz33HMnju/t7Z0pSLSzsxObm5uxv79/eGx1dTW2t7djfX195PcbRpqm0Wq1olarnVrzKNc4b72vXr0ab7755lT2RrfepaWlODg4GPuePLp2ETH0Ok7Kzs5OPP/88z1rfvXq1fjRj350pv1z586dePnll3uOnWd/T8skn0V5z4LT1uT4fTbKfTeMfs+Oa9euRafT6Rl9Os3rN+5zhIiz34MAAAAAAEVJkuRulmVr/V7TyQwAuHDmYXxVv+4nEREPHjyIdrsdL730Uu455HWrevjw4Vg7qkyr+9ZRw3bXGvUa91vvz3zmM/Htb3974KjScXbDO9qV6u7du/GLv/iLJ973tM/Le/14N6tuELDIkEO/fZIkyUj75+j53rx5M954441YXl6OlZWVobvdFWnSz6KzdlE6fp+Nu6tdv2dHp9MprOPTODo8Qj9n7cIJAAAAADCLhMwAgAtnHsZX9QthHHX58uW4dKn3V7mj59BvjN64Q2Gz/Mfzftcyy7Lca9xvbT788MN47bXXckMnecGU8wTPKpVK/P3f/30899xzJ973tCDMoHqGDTJNc4TsefdPs9mMZ555Jr7whS/EM888E81mM27evBk/+clP4u233x5plGlRJv0sKiIIOox+1/7111+Px48f93zdNGqdh9Ax8+2sI5YhorjR7gAAAADQj3GZAMCFM+7xVZMas9Ydo3f58uV4+PBhz2tXr16NJElGPofzjmTsd66zOGbunXfeic997nMnjv/d3/1dXL9+ve/3dNfmypUr8eDBg57Xjq9t3h66fft23Lp168yjD8866nTQnm61WkONNS1qhOxZ9k+apvHzP//zcXBwcHhsaWkp3nvvvZnZg8OYxii9SY9hPY/j176IWosY+wswjFkf7Q4AAADAYjIuEwDgiGE7KA3TPWKSY9a63U/efvvteOONN3rqffPNN8/UBeo8HVXyznXco/TG4eHDh1Eul3uOlcvlE2G9o7a2tuLu3btx69ateOqpp3peO95dql8HqitXrsRv/MZvnKsjUl5nqx//+McDO14N6og1TDerIrs5nWX/3L9/vydgFvHJOd2/f3/c5U3UNLoBDnvPT7JbTt57H7/2RXR8mtVub8DFpssiAAAAALPoStEFAAAUYWtrKzY2NnI7KA3TPeLoHwC7nYjq9XpsbGyMLSRSqVQOgxhf+cpXTtQ76ByO13r060atbxrnOi5pmsb7778f/Tr2DgqOHO1k9sEHH/S8djx00i+Y8ujRoxPBp27Qa9g1ygu8fP7znx8YhBkUlOkGmY53iDpaUzekdrSj1qi1z6J56Lx32rNoHE675yfZLWfU9z7L8+k8hrk/AKZtUX8uAwAAADDfjMsEADhm2BF28zJmbRwBkrxz/f73vx+f/exnzxSOmUTY5+i5ttvtyLIsyuXyqaP3+l3ziIiVlZV48uRJz/d26753717cunUrrly5Ep1OJ548eRKPHz/u+f6r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\n",
      "text/plain": [
       "<Figure size 7200x7200 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Lets see what this looks like, just for fun:\n",
    "# Plot the original data and the PCs\n",
    "fig = plt.figure(figsize=(100,100))\n",
    "ax2 = fig.add_subplot(222)\n",
    "ax2.scatter(X_PCA[:, 0], X_PCA[:, 1], c='black', s=20, edgecolor='k')\n",
    "plt.axis('scaled')\n",
    "plt.xlabel('PC1') \n",
    "plt.ylabel('PC')\n",
    "plt.title('Principal Components')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Very interesting!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.00249711 0.00182072 0.00161226 0.00139908 0.00132104 0.0012607\n",
      " 0.00115983 0.00112929 0.00110076 0.00109272]\n"
     ]
    }
   ],
   "source": [
    "expl_var = my_pca.explained_variance_ratio_\n",
    "print(expl_var)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(range(0,10), expl_var, marker='*')\n",
    "plt.xlabel('Number of PCs')\n",
    "plt.ylabel('Proportion of Variance Explained')\n",
    "plt.xticks(range(0,10,1))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using a PC of 7 gets us much better accuracy results.\n",
    "my_pca = PCA(n_components=7)\n",
    "my_pca.fit(X_train)\n",
    "\n",
    "X_PCA_train = my_pca.transform(X_train)\n",
    "X_PCA_test = my_pca.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "BEFORE PCA:\n",
    "- The train accuracy is: 0.811\n",
    "- The test accuracy is: 0.785"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The train accuracy is: 0.771\n",
      "The test accuracy is: 0.761\n"
     ]
    }
   ],
   "source": [
    "# AFTER PCA:\n",
    "# Instantiate and train the classifier\n",
    "logistic_regression_model = LogisticRegression(solver='lbfgs', max_iter=100000)\n",
    "logistic_regression_model.fit(X_PCA_train, y_train)\n",
    "# Evaluate it\n",
    "\n",
    "print(f'The train accuracy is: {logistic_regression_model.score(X_PCA_train, y_train):0.3f}')\n",
    "print(f'The test accuracy is: {logistic_regression_model.score(X_PCA_test, y_test):0.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- That was SO much faster!!! Though our accuracy suffered..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5.56 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%%timeit -r 1\n",
    "my_pca = PCA(n_components=1)\n",
    "my_pca.fit(X_train2)\n",
    "X_PCA_train = my_pca.transform(X_train2)\n",
    "X_PCA_test = my_pca.transform(X_test2)\n",
    "logistic_regression_model = LogisticRegression()\n",
    "logistic_regression_model.fit(X_train2, y_train2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There seems to be an exponential increase as the n_components increases but not until nears 1000 components\n",
    "- n_components=1 : 5.57 s ± 282 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n",
    "- n_components=10 : 5.56 s ± 315 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n",
    "- n_components=100 : 6.67 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)\n",
    "- n_components=1000 : 18.3 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)\n",
    "- n_components=2000 : 41 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 1.4. List one advantage and one disadvantage of dimensionality reduction\n",
    "- Advantage: We can get nearly the same LogReg results from 20 dimensional data as we can from 2177 dimensional data and it is MUCH faster.\n",
    "- Disadvantage: Your independant variables become less interpretable and therefore not as readbale as before."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 2.1 Fit a KNN model to this data. What is the accuracy score on the test set?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.neighbors import KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting up our variables\n",
    "y_train = df_train['Reviewer_Score']\n",
    "X_train = df_train.drop(['Reviewer_Score'], axis=1)\n",
    "y_test = df_test['Reviewer_Score']\n",
    "X_test = df_test.drop(['Reviewer_Score'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set accuracy: 0.642543217111046\n"
     ]
    }
   ],
   "source": [
    "# Instantiate the model & fit it to our data\n",
    "KNN_model = KNeighborsClassifier(n_neighbors=3, weights='distance')\n",
    "KNN_model.fit(X_train, y_train)\n",
    "\n",
    "# Score the model on the test set\n",
    "test_predictions = KNN_model.predict(X_test)\n",
    "test_accuracy = accuracy_score(test_predictions, y_test)\n",
    "print(f\"Test set accuracy: {test_accuracy}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Test set accuracy: 0.6428362144740697"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 2.2 KNN is a computationally expensive model. Reduce the number of observations (data points) in the dataset. What is the relationship between the number of observations and run-time for KNN?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Our data frame is quite large! Let's sample a portion of it\n",
    "X_train_small = X_train.sample(frac=.1, random_state=42)\n",
    "y_train_small = y_train.sample(frac=.1, random_state=42)\n",
    "X_test_small = X_test.sample(frac=.1, random_state=42)\n",
    "y_test_small = y_test.sample(frac=.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set accuracy: 0.642543217111046\n",
      "Test set accuracy: 0.642543217111046\n",
      "6.66 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%%timeit -r 1\n",
    "# Instantiate the model & fit it to our data\n",
    "KNN_model = KNeighborsClassifier(n_neighbors=3, weights='distance')\n",
    "KNN_model.fit(X_train, y_train)\n",
    "\n",
    "# Score the model on the test set\n",
    "test_predictions = KNN_model.predict(X_test)\n",
    "test_accuracy = accuracy_score(test_predictions, y_test)\n",
    "print(f\"Test set accuracy: {test_accuracy}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- 6.84 s ± 78.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set accuracy: 0.5835777126099707\n",
      "Test set accuracy: 0.5835777126099707\n",
      "324 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%%timeit -r 1\n",
    "# Instantiate the model & fit it to our data\n",
    "KNN_model = KNeighborsClassifier(n_neighbors=3, weights='distance')\n",
    "KNN_model.fit(X_train_small, y_train_small)\n",
    "\n",
    "# Score the model on the test set\n",
    "test_predictions = KNN_model.predict(X_test_small)\n",
    "test_accuracy = accuracy_score(test_predictions, y_test_small)\n",
    "print(f\"Test set accuracy: {test_accuracy}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- 355 ms ± 15.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There is a strong relationship between the number of observations and the runtime for KNN. Reducing the observations by a factor of 10 brought almost a 20 fold decrease in runtime!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 2.3 List one advantage and one disadvantage of reducing the number of observations.\n",
    " - Advantage: Decreases KNN Runtime.\n",
    " - Disadvantage: Decreases Test-set Accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 2.4 Use the dataset to find an optimal value for K in the KNN algorithm. You will need to split your dataset into train and validation sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "#Let's reset our variables\n",
    "y_train = df_train['Reviewer_Score']\n",
    "X_train = df_train.drop(['Reviewer_Score'], axis=1)\n",
    "y_test = df_test['Reviewer_Score']\n",
    "X_test = df_test.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "X_train = X_train.sample(frac=.1, random_state=42)\n",
    "y_train = y_train.sample(frac=.1, random_state=42)\n",
    "X_test = X_test.sample(frac=.1, random_state=42)\n",
    "y_test = y_test.sample(frac=.1, random_state=42)\n",
    "\n",
    "# Train-Test-Split the data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)\n",
    "\n",
    "#Transform data\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train = scaler.transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set accuracy: 0.6952879581151833\n",
      "Test set accuracy: 0.4658536585365854\n"
     ]
    }
   ],
   "source": [
    "# Fit the default KNN model to this data.\n",
    "#Evaluate the model's accuracy.\n",
    "\n",
    "# 1. Instantiate\n",
    "my_knn = KNeighborsClassifier(n_neighbors = 2)\n",
    "# 2. Fit\n",
    "my_knn.fit(X_train, y_train)\n",
    "\n",
    "# 3. Predict & Evaluate\n",
    "train_accuracy = my_knn.score(X_train, y_train)\n",
    "test_accuracy = my_knn.score(X_test, y_test)\n",
    "\n",
    "print(f\"Test set accuracy: {train_accuracy}\")\n",
    "print(f\"Test set accuracy: {test_accuracy}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "955"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Finished 19 estimators and counting...\r"
     ]
    }
   ],
   "source": [
    "k_values = list(range(1, 20+1, 2))\n",
    "train_accuracies = []\n",
    "test_accuracies = []\n",
    "\n",
    "for k in k_values:\n",
    "    # 1. Instantiate\n",
    "    my_knn = KNeighborsClassifier(n_neighbors = k)\n",
    "    # 2. Fit\n",
    "    my_knn.fit(X_train, y_train);\n",
    "    \n",
    "    # 3. Predict & Evaluate & append (save)\n",
    "    train_accuracy = my_knn.score(X_train, y_train)\n",
    "    test_accuracy = my_knn.score(X_test, y_test)\n",
    "    train_accuracies.append(train_accuracy)\n",
    "    test_accuracies.append(test_accuracy)\n",
    "    print(f'Finished {k} estimators and counting...', end='\\r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(k_values, train_accuracies, c='blue', label='train')\n",
    "plt.plot(k_values, test_accuracies, c='red', label='test')\n",
    "plt.xlabel('N Neighbors')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- optimal k seems to exist between 7.5 and 10"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 2.5 What is the issue with splitting the data into train and validation sets after performing vectorization?\n",
    "- This is an issue because once vectorized, you have preformed preprocessing which carries information from the train set into the validation set."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 3.1 Fit a decision tree model to this data. What is the accuracy score on the test set?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DT training set accuracy: 0.9958115183246073\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "#Setting up our variables\n",
    "y_train = df_train['Reviewer_Score']\n",
    "X_train = df_train.drop(['Reviewer_Score'], axis=1)\n",
    "y_test = df_test['Reviewer_Score']\n",
    "X_test = df_test.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "#Sub-sampling our data\n",
    "X_train = X_train.sample(frac=.1, random_state=42)\n",
    "y_train = y_train.sample(frac=.1, random_state=42)\n",
    "X_test = X_test.sample(frac=.1, random_state=42)\n",
    "y_test = y_test.sample(frac=.1, random_state=42)\n",
    "\n",
    "# Train-Test-Split the data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)\n",
    "\n",
    "# Instantiate & fit the DT\n",
    "DT_model = DecisionTreeClassifier(max_depth=20)\n",
    "DT_model.fit(X_train, y_train) \n",
    "\n",
    "# Evaluate its classification accuracy (Just on the training set for now)\n",
    "print(f\"DT training set accuracy: {DT_model.score(X_train, y_train)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Finished 50 estimators and counting...\r"
     ]
    }
   ],
   "source": [
    "max_depth = list(range(1, 50+1, 1))\n",
    "train_accuracies = []\n",
    "test_accuracies = []\n",
    "\n",
    "for k in max_depth:\n",
    "    # Instantiate & fit the DT\n",
    "    DT_model = DecisionTreeClassifier(max_depth=k)\n",
    "    DT_model.fit(X_train, y_train)\n",
    "\n",
    "    # Evaluate its classification accuracy (Just on the training set for now)\n",
    "    train_accuracies.append(DT_model.score(X_train, y_train))\n",
    "    test_accuracies.append(DT_model.score(X_test, y_test))\n",
    "    print(f'Finished {k} estimators and counting...', end='\\r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(max_depth, train_accuracies, c='blue', label='train')\n",
    "plt.plot(max_depth, test_accuracies, c='red', label='test')\n",
    "plt.xlabel('Max Depth')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xticks(range(0,50,2))\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Having a max_depth value of 8 seems to produce the highest test accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 3.3 Provide two advantages of decision trees over KNN. Provide two weaknesses of decision trees (classification or regression trees)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Like KNN, decision trees are another example of supervised learning:\n",
    "- Advantage 1: Can learn highly non-linear decsion boundaries.\n",
    "- Advantage 2: Can handle multi class data.\n",
    "- Disadvantage 1: Decision trees are constrained to notions of closeness.\n",
    "- Disadvantage 2: Prone to overfitting!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 4. What is the purpose of the validation set, i.e., how is it different than the test set?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The validation set is used to adjust hyperparameters. The test set is kept seperate as we don't want it to interact with the model so that we can have an unbiased estimate of our model's accuracy. We use the train and validation sets to guide the model. Once tuned, then we then use the test set to test the model against a \"fresh\" set of data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 5. Re-run a decision tree or logistic regression on the data again:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 5.1. Perform a 5-fold cross validation to optimize the hyperparameters of your model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "# Filter warnings\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting up our variables\n",
    "y_train = df_train['Reviewer_Score']\n",
    "X_train = df_train.drop(['Reviewer_Score'], axis=1)\n",
    "y_test = df_test['Reviewer_Score']\n",
    "X_test = df_test.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "#Sub-sampling our data for speedy test and checks\n",
    "#Will toggle on an off as needed to increase accuracy\n",
    "X_train = X_train.sample(frac=.1, random_state=42)\n",
    "y_train = y_train.sample(frac=.1, random_state=42)\n",
    "X_test = X_test.sample(frac=.1, random_state=42)\n",
    "y_test = y_test.sample(frac=.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting out 20% for test set\n",
    "X_remainder, X_test, y_remainder, y_test = train_test_split(X_train, y_train, test_size = 0.2,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.72146119 0.71689498 0.75229358 0.68348624 0.70183486]\n"
     ]
    }
   ],
   "source": [
    "# 1. Instanitate model\n",
    "my_logreg = LogisticRegression(random_state=1)\n",
    "\n",
    "# 2. Fit model on 5 folds.\n",
    "# The variable \"scores\" will hold 5 accuracy scores, \n",
    "# each from a different train and validation split\n",
    "scores = cross_val_score(my_logreg, X_remainder, y_remainder, cv = 5)\n",
    "print(scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average Score:0.7151941686565289\n"
     ]
    }
   ],
   "source": [
    "print(f\"Average Score:{np.mean(scores)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Finished cross validation with c = 100000000000.0..\r"
     ]
    }
   ],
   "source": [
    "#Store the results\n",
    "cross_validation_scores = []\n",
    "\n",
    "C_range = np.array([.00000001,.0000001,.000001,.00001,.0001,.001,.1,\\\n",
    "                1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000,10000000000,100000000000])\n",
    "\n",
    "#Do some cross validation\n",
    "for c in C_range:\n",
    "    LR_model = LogisticRegression(C=c,random_state=1)\n",
    "    \n",
    "    # the cross validation score (mean of scores from all folds)\n",
    "    cv_score = np.mean(cross_val_score(LR_model, X_remainder, y_remainder, cv = 5))\n",
    "    \n",
    "    cross_validation_scores.append(cv_score)\n",
    "    print(f'Finished cross validation with c = {c}..', end='\\r')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYgAAAEKCAYAAAAIO8L1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3deXxU1dnA8d+TjbDvCUuAgALKvoTFDYJbqbgUbRXUKlq1rcVarbb6tq9S27611qpVUYsWaxWFakVRUEqpQVAQCBBkX8KSsIaEBBIIWeZ5/7g3cQiTyWSZzCQ8389nPpm5c+69zx2G+8w959xzRFUxxhhjKooIdQDGGGPCkyUIY4wxPlmCMMYY45MlCGOMMT5ZgjDGGOOTJQhjjDE+RYU6gLrSoUMHTUxMDHUYNVZQUEDz5s1DHUaNWfyhZfGHVkOOPzU19YiqdvT1XqNJEImJiaxevTrUYdRYSkoKycnJoQ6jxiz+0LL4Q6shxy8ieyp7z6qYjDHG+GQJwhhjjE+WIIwxxvhkCcIYY4xPliCMMcb4ZAnCGBNSqXuO8vHOIlL3HA11KKYCSxDGmJBJ3XOUm19dwb+2F3PzqyssSYSZRnMfhDGm4ThWWMy8dft54b/bOVXiAeBUiYfvv/YVA7q25py4FpzTsTnnxLXg3I4t6NqmKREREuKozz6WIIwx9UJVWbP3KO+szGD++gOcLC6lR/tmREUIpR4lMkK48Nz25J0sZuHGg+QUFJWvGxsdQc8OLTi3LHF0dJ737NCcjfuPsSI9m9G92jO8R9sQHmHNpO45Wqv4a7u+P5YgjDFBdbSgiPfX7mP2yr1sP5xP85hIvjO0C5NGdGdQQmvW7M3lnf+sYvLlI047weUUFLEzK5+dh/PZcTifnVn5pGXk8vH6/XhPhCmAAhECV/SLp298S9o2j6Fd8xjaNnP/No+hXbMYmsZEnhFfME+wFakqR/KLyo9n+c4jfLLhIB4FETinQ3OaNwn8tFxwqoSdRwpQdZLorLtG1+kxWIIwxtQ5j0dZkZ7NO6syWLjhIEWlHoZ0a8MfbxjI1YO6nHYSHN6jLcfPiTnjxNaueQztmrdjRGK705YXFpey60gBO7PyefurvXy5M9vZp8LSbUdYtOkQnkpmUo6NjqBds5jyBKIKy9Oz8bhXMD+9rDdJiW2dfTeLoU2zGGKi/DfVljWyt+x5tPwYSko9ZBw9yU43Eezw+nussKR83agIKY9VFUoV2jaPCegzBjhWWFKeLItLPKxIz7YEYYy3+ev3s/1wPpf07tggqxgak8PHC3kvNZM5qzLYk32CVrFR3DyqO5NGduO8Tq3qZB+x0ZGc37kV53duRefWTVnz2gqKSzxER0Xw5l2jGNqtDXkni8k5UcTRgiJyCoo4eqKInIJi96+7/EQRu44UUOqeoUs8yjOLtp2xv5axUeVXI+2bx5x2dZJfWMyMpemUlCof7FzOiMS2ZBcUsfvICYpKPeXb6NCiCefGNeeawV3carIWnBPXggO5J7n1b1+Vx//09wZX6zucuucot3gd/+he7Wv/AXuxBGEaJFVl5a4cnvx0C2v35gLwcspO3r67bi+xTdVW7c5h9sq9ZOScIHVvLqUeZVTPdjxweR/GD+hEbPSZ1Tp1ZXiPtsy6a/QZVURt3RM5Psco/Yb3CTYqMoInrx9Ep9ax5LgJpCzBlCWZg8cK2XTgGNkFRRSVeE7bVolH2XLwOMN7tGXceXHl7STndGhB62bRPvfftU1Tn/HX9vjriiUI06B4PMp/txzm5SU7Sd1zlGbRkeV10KdKPLzx5W5LEEFSVOJhT7ZTtbMzq4Cdh/NZn5nLjqyC8jLXDenC/Zf1plfHFvUW1/AebWv8b17TE6yqcrK4lCVbs7h/zjqKSzw0iY7gtdtH1OgkX5vvbG3X98cShGkQiks9fLx+P6+kpLP10HG6tmnKE9f159yOLbjzjVUUl3jwAPPS9hPfqgm/HH8eUZF2m08gKjbS5p4ochKA20BclhD25pwor44B6Nw6lhivzzhSoE98y3pNDnWhJidYEaFZTBTfHtiZuFaxPhvZG4OgJggRGQ/8BYgEXlPVJyu8/ywwzn3ZDIhT1TYiMgR4GWgFlAK/V9U5wYzVhKfC4lL+uTqDGZ+nk3n0JH3iW/DsTYO5elAXot2TU9kvwKQebZn/9QFeXbqLr/fl8eLNw+jQokmIjyC8pe7OYdKrKyguVSLEqW/PO/lNI2pMZAQ9OzTn/M4tuXpQZ6fuvGMLenZsTosmUUGvA28IKmtkbwyCliBEJBKYDlwBZAKrRGSeqm4qK6OqD3iVvw8Y6r48AdymqttFpAuQKiILVTU3WPGa8JJ3spi3Vuxh5rJdZBcUMax7G6Zd059Lz4s744Yp71+Ao3q1Z3BCG/5n7tdc/fwyXrp1GMO6N77/uHVhZ1a+Uz1S6lwVeBR6tHMaUs+Jc+41SGjbjEg/N6gFuw7chFYwryBGAjtUNR1ARGYD1wGbKik/GXgcQFXLuxKo6n4ROYzT3GQJopE7fKyQv32xi1kr9pJ/qoTkvh358dhzGNmzHSKB3Ul7w/AEzuvckh+9lcpNf13O49f055ZR3QNev7rqsx99XThVUsrLKTt56bOdREUK0ZGCx6NER0Xw+LX9670O3YQvUa2kw3BtNyzyXWC8qt7lvv4+MEpVp/oo2wNYASSoammF90YCbwD9VdVT4b17gHsA4uPjh8+ePTsox1If8vPzadGiYdXdeqtt/IdPeFiwq5hl+0oo9cDITpFc1SuaHq1q3gMmv0iZsf4U64+UclGXKG7vH0NMpO8kUZP4s054+GBnEV/sc76ykQI/H96Efh3qv2kv0Pi35pTy942nOFCgjOoUyeTzYzhyQtmSU8p57SI5t23wehz5c7Z//0Np3Lhxqaqa5Ou9YH6Tff1PrCwbTQLe85EcOgNvArdXTA4AqjoDmAGQlJSkDXVOWGjYc9pCzePfuD+PV5akM3/9fqIiIrhxRHfuuaQXiR3qZgL4qy5Xnlu8necXbyeXZrxy63C6tWt2RrlA4y8q8fCfzYd4Z+Velu04ctodvaUKz60t5obh8Uwe2Y2BXVsH7aqloqrizztRzB8+2czsVRkktG3K328cQHLfuHqJLRBn6/c/3AUzQWQC3bxeJwD7Kyk7CfiJ9wIRaQXMB36tqiuCEqEJibJ7GF5espOUrVm0aBLF3WN68YOLehLXKrZO9xURITx4RR8GJ7TmgTnruObFZfxl0lDG9qmig3wFO7PymbMqg3+lZpJdUESX1rH89NLe9OvSivtnr6W4xENkRAQXntOeuWszeWflXvp1bsXkkd24bmhXWsX67gcfbKrKvLT9/PbjTRw9UcwPx/Ti/st70yzGOjCaqgXzW7IK6C0iPYF9OEng5oqFRKQv0BZY7rUsBpgL/ENV3w1ijKYeVbyHoX3zGB7+Vl9uHd2D1k2DewK97Px4PrrvYn74ZipTXl/Jz6/ow73J5/odIbSwuJRPNhzgnZUZrNyVQ1SEcNn5cUwa2Z0xvTuWN95WbKTNO1nMvHX7eGdlBv/74UZ+v2AzEwZ2YdLIbiT1aFtvVxUZOSf41Qcb+HxbFoMTWvPGnSPp36V1vezbNA5BSxCqWiIiU4GFON1cZ6rqRhF5AlitqvPcopOB2Xp6Y8iNwBigvYhMcZdNUdV1wYrXBE9l9zB8b3g3n4OnBUuP9s2Ze+9FPPr+ep7+9zbWZeTx5xsHn5GcNh84xuyVe5m7dh/HCkvo0b4Zvxjfl+8OTyCu5ZlXOBUbaVs3jeb7FyRy6+gebNh3jHdW7WXeuv38a00m58a1YNKIblw/LIF21RhzpzqKSz3MXLaLZ/+zjUgRHr+mH7ddkOi3N5IxvgT1OlNVFwALKix7rMLraT7Wewt4K5ixmeAL5B6G+tY0JpJnbxrCkG5t+N38zVz34jLuv6wPKduLWHFyM8vTc0jLyCUmMoLxAzoxaWQ3RvdsX6O5CESEgQmtGZgwkF9ddT7z1x9g9qq9/G7+Zv746Rau7N+JySO6ExsdwVe7cuqkF1RaRi6PvP81mw8c44p+8fzm2v50adO0Vts0Zy+riDR1rjr3MISCiDDlop7079qau/+xmgf+6V6Y7kwnoW1T/vfqflw/tGu1RtWsSvMmUdw4ohs3jujG1oPHmb1qL++v2cf89QfKhwqJjBAmjXAat9s1P32Y6tZNoyv97FL3HGXu9iL+uS+VTzYcJK5lE165dTjjB3Sqs/jN2ckShKkzdXEPQ30akdiOm0d256WUnYAzn8Dkkd34wcU9g7rfvp1a8vg1/fnl+PN46N00Pl5/AIBSjzLrq70+14kQaOPObeAMVx1Nu+YxnCr2MC9tPyUeBQ7y7QGdeOq7g2gZokZx07hYgjC1tmD9Af60/CQZi/6Lx6NMGNSFH43t1SAaRC87P56ZX+yiqNhDTFQEo3t1qLd9x0ZHcsdFPfnP5kPlQ1XMvH0EPTo052hBEdkFFYerdv5m5zvDVKfuySW74FR5V9sIgQFdW1tyMHXGEoSplY/X72fq22sBp4rkhclDmTCoS4ijClzZUBGhGmytsqEqugbYbpC6O4dbXvuKopKyBHf2jYVkgscShKmx4lIPv/vYa+QUVXZnnwhdQDUU6sHWajVcdWI7Zt0dugRnGjdLEKbGnl+8nYPHThEdKZSW6lk7mmeohTrBmcbLEoSpkVW7c5j+2Q6+OzyBySO72y9YYxohSxCm2o4VFvPAnHUktG3GtGv706JJlP2CNaYRsgRhqu3xDzdyIK+Qd390AS2a2FfImMbK5mQ01TIvbT9z1+7jvkvPtYl4jGnkLEGYgO3LPcmv5n7NsO5tmDru3FCHY4wJMksQJiClHuXBOevweJTnbhpKVIjGUjLG1B+rQDYBmfF5Ol/tyuFP3x1E9/ZnTrhjjGl87GegqdKGfXk8s2grVw3sxHeHJ4Q6HGNMPbEEYfw6WVTKT2evpX3zJvzfxIFhOeieMSY4rIrJ+PW7+ZvYdaSAWT8YRZtmwZngxhgTnuwKwlTqP5sOMeurvdx9SS8uPLf+Rjk1xoQHSxDGp6zjp/jlv9bTr3Mrfn5ln1CHY4wJAatiMmdQVX7xXhr5p0qYPWkITaLqb95oY0z4COoVhIiMF5GtIrJDRB7x8f6zIrLOfWwTkVyv924Xke3u4/ZgxmlO9+aKPXy2NYv/uep8ese3DHU4xpgQCdoVhIhEAtOBK4BMYJWIzFPV8gkEVPUBr/L3AUPd5+2Ax4EknOl6U911jwYrXuPYfug4v5+/meS+Hbntgh6hDscYE0LBvIIYCexQ1XRVLQJmA9f5KT8ZeMd9/i1gkarmuElhETA+iLEa4FRJKffPXkfzJlE89d1B1qXVmLNcMNsgugIZXq8zgVG+CopID6An8F8/63b1sd49wD0A8fHxpKSk1DroUMnPzw95/HO2FrHpQDH3D2vCptQVbKp6lXLhEH9tWPyhZfGHp2AmCF8/P7WSspOA91S1tDrrquoMYAZAUlKSJicn1yDM8JCSkkIo4/9yxxE+XfgVt4zqzgMTB1Z7/VDHX1sWf2hZ/OEpmFVMmUA3r9cJwP5Kyk7im+ql6q5ramnJ1sPc82YqnVvH8usJ/UIdjjEmTAQzQawCeotITxGJwUkC8yoWEpG+QFtgudfihcCVItJWRNoCV7rLTB1L3XOUO/6+ivxTJRzJL2LTgWOhDskYEyaCliBUtQSYinNi3wz8U1U3isgTInKtV9HJwGxVVa91c4Df4iSZVcAT7jJTx/696SAe95MvLfWwIj07tAEZY8JGUG+UU9UFwIIKyx6r8HpaJevOBGYGLTgDgLjJIUIgOiqC0b3ahzYgY0zYqDJBiNPX8Ragl6o+ISLdgU6qujLo0Zmg25VdQIcWMdxxUSKje3VgeA+bRtQY4wjkCuIlwANcCjwBHAf+BYwIYlymHhSXevhiRzbXDunCT8b1DnU4xpgwE0iCGKWqw0RkLYCqHnUbnU0Dl7rnKPmnShjbp2OoQzHGhKFAGqmL3WEzFEBEOuJcUZgGbsm2LKIihAvPsXYHY8yZAkkQzwNzgTgR+T2wDPi/oEZl6kXK1iySEtvSMjY61KEYY8JQlVVMqjpLRFKBy3DucP6Oqm4OemQmqA4dK2TzgWP8cvx5oQ7FGBOm/CYIEYkA1qvqAGBL/YRk6sPn27IArP3BGFMpv1VMquoB0tyuraYRSdmWRVzLJpzf2eZ7MMb4Fkgvps7ARhFZCRSULVTVaytfxYSzklIPy7Yf4cp+8TaktzGmUoEkiN8EPQpTr9Iyc8k7WUxy37hQh2KMCWOBNFIvEZF4vrkxbqWqHg5uWCaYlmzNIkLg4nM7hDoUY0wYq7Kbq4jcCKwEvgfcCHwlIt8NdmAmeJZsy2Jo97a0bmbdW40xlQukiulXwIiyqwb3Rrn/AO8FMzATHNn5p1i/L48HL+8T6lCMMWEukBvlIipUKWUHuJ4JQ0u3H0EVxva17q3GGP8CuYL4VEQW8s2MbzcBnwQvJBNMKVsP0755DAO6tA51KMaYMBdII/XDInI9cDHOndQzVHVu0CMzdc7jUT7ffoSxfToSEWHdW40x/gUyH0RPYIGqvu++bioiiaq6O9jBmbq1YX8eOQVFdve0MSYggbQlvMvpo7eWustMA5OyNQsRuKS3dW81xlQtkAQRpapFZS/c5wHNByEi40Vkq4jsEJFHKilzo4hsEpGNIvK21/Kn3GWbReR5sVt+a23JtiwGdW1N+xZNQh2KMaYBCCRBZIlI+bAaInIdcKSqldw5JKYD3wb6AZNFpF+FMr2BR4GLVLU/8DN3+YXARcAgYADOTXpjAzkg41veiWLW7j1q1UvGmIAF0ovpR8AsEXkRp5E6A7gtgPVGAjtUNR1ARGYD1wGbvMrcDUxX1aMAXt1pFYjFuVIRIBo4FMA+TSWW7sjCozDWhtcwxgQokF5MO4HRItICEFU9HuC2u+IkkzKZwKgKZfoAiMgXQCQwTVU/VdXlIvIZcAAnQbzoaw4KEbkHuAcgPj6elJSUAEMLP/n5+UGNf87Xp2geDXnp60jZVfe1dcGOP9gs/tCy+MNTpQlCRK7BmQtij7voQeAGEdkD3K+qu6rYtq+zkPrYf28gGUgAlorIAKADcL67DGCRiIxR1c9P25jqDGAGQFJSkiYnJ1cRUvhKSUkhWPGrKr/4YjHjzu/IpeOGBWUfwYy/Plj8oWXxhyd/bRC/B7IARORq4FbgTmAe8EoA284Eunm9TgD2+yjzoaoWuwlnK07CmAisUNV8Vc3HuTFvdAD7ND5sPnCcw8dPWfuDMaZa/CUIVdUT7vPrgb+paqqqvgYEcqZZBfQWkZ4iEgNMwkku3j4AxgGISAecKqd0YC8wVkSiRCQap4HapjmtoSU2e5wxpgb8JQgRkRbutKOXAYu93outasOqWgJMBRbinNz/qaobReQJr15RC4FsEdkEfAY8rKrZOAMB7gS+BtKANFX9qJrHZlwpWw/Tr3Mr4lpV+c9mjDHl/DVSPwesA44Bm1V1NYCIDMVpPK6Sqi4AFlRY9pjXc8Vp23iwQplS4IeB7MP4d7ywmNQ9R7l7TK9Qh2KMaWAqTRCqOtMdpC8O51d8mYPAHcEOzNSNL3ZkU+JRq14yxlSb326uqroP2FdhWUBXDyY8LNmWRYsmUQzv0TbUoRhjGhib16ERU1U+35bFRee2JzrS/qmNMdVjZ41GbMfhfPblniTZ7p42xtRAIENtlI2rFO9dXlX3BisoUzfKureOsfYHY0wNBDIfxH3A4zhjIZUN+604A+mZMJayNYvecS3o2qZpqEMxxjRAgVxB3A/0de9PMA3EiaISVu7K4fYLe4Q6FGNMAxVIG0QGkBfsQEzdWpGeTVGph7F9rP3BGFMzgVxBpAMpIjIfOFW2UFWfCVpUptZStmbRNDqSET2te6sxpmYCSRB73UcMAc4kZ0JvybYsLjynPU2iIkMdijGmgQpkPojfAIhIS+el5gc9KlMru44UsCf7BD+4uGeoQzHGNGBVtkGIyAARWQtsADaKSKqI9A9+aKamlmx1JuZLtvYHY0wtBNJIPQN4UFV7qGoP4OfAq8ENy9TGkm1Z9OzQnO7tm4U6FGNMAxZIgmiuqp+VvVDVFKB50CIytVJYXMry9GwbnM8YU2sB9WISkf8F3nRf3wpUNd2oCZGVu3IoLPYwtq8lCGNM7QRyBXEnzgxy7wNz3ec23HeYStmaRUxUBKN7tg91KMaYBi6QXkxHgZ/WQyymDizZdpjRvdrTNMa6txpjaqfSBCEiz6nqz0TkI5yxl06jqtf6WM2EUEbOCXZmFXDzKBtewxhTe/6uIMraHJ6uj0BM7ZWN3moN1MaYulBpG4SqprpPh6jqEu8HMCSQjYvIeBHZKiI7ROSRSsrcKCKbRGSjiLzttby7iPxbRDa77ycGflhnpyXbskho25RzOlonM2NM7QXSSH27j2VTqlrJnUNiOvBtoB8wWUT6VSjTG3gUuEhV+wM/83r7H8CfVPV8YCRwOIBYz1pFJR6+3HGEsX06IiKhDscY0wj4a4OYDNwM9BSReV5vtQQCGfp7JLBDVdPd7c0GrgM2eZW5G5juNoSjqofdsv2AKFVd5C634T2qsHpPDgVFpTZ7nDGmzvhrg/gSOAB0AP7stfw4sD6AbXfFGSq8TCYwqkKZPgAi8gUQCUxT1U/d5bki8j7QE/gP8IiqlnqvLCL3APcAxMfHk5KSEkBY4Sk/P79W8f9zaxGRAiX7N5FyeHPdBRag2sYfahZ/aFn84anSBKGqe4A9wAU13Laveo6KvaGigN5AMpAALBWRAe7yS4ChOCPJzsGp1vpbhRhn4AwFQlJSkiYnJ9cw1NBLSUmhNvE/ue5zRvZsxbcvH113QVVDbeMPNYs/tCz+8BTIYH2jRWSViOSLSJGIlIrIsQC2nQl083qdAOz3UeZDVS1W1V3AVpyEkQmsVdV0VS0BPgCGBXJAZ6ODeYVsOXicZLt72hhThwJppH4RmAxsB5oCdwEvBLDeKqC3iPQUkRhgEjCvQpkPgHEAItIBp2op3V23rYiUnfEu5fS2C+Pl87LurZYgjDF1KJAEgaruACJVtVRVX8c9qVexTgkwFVgIbAb+qaobReQJESm7yW4hkC0im4DPgIdVNdtta3gIWCwiX+NUV9kIspWYuzaTFk2iKCgsCXUoxphGJJDB+k64VwDrROQpnIbrgDraq+oCYEGFZY95PVfgQfdRcd1FwKBA9nM2W7U7h+XpOQDc8revmHXXaIb3sGlGjTG1F8gVxPdxehhNBQpw2hVuCGZQJnDz1x8of15c4mFFeiA9kI0xpmqBDNa3x316EvhNcMMx1dXcHZQvQiA6KoLRvWwUV2NM3fB3o9zX+Bikr4yqWvVPGMg9WUyz6EjuHXcOF5zTwaqXjDF1xt8VxNXu35+4f8sG77sFOBG0iEy1pGXmMrRHG6Ze2jvUoRhjGhl/g/XtcauXLlLVX6jq1+7jEeBb9ReiqUxhcSlbDhxnUEKbUIdijGmEApqTWkQuLnshIhdic1KHhc0HjlHiUQZbgjDGBEEg3Vx/AMwUkdbu61ycaUhNiKVl5AIwuFvrKkoaY0z1BdKLKRUYLCKtAFHVvOCHZQKxPjOPji2b0KlVbKhDMcY0Qv56Md2qqm+JyIMVlgOgqs8EOTZThbTMXAYntLH5H4wxQeGvDaKsnaFlJQ8TQscKi9mZVcDgBKteMsYEh7/hvv/q/rWb48LQhkynpm9QN2ugNsYEh78qpuf9raiqP637cEyg0twEYVcQxphg8ddInVpvUZhqS8vIpUf7ZrRpFhPqUIwxjZS/KqY36jMQUz3rM3MZntgu1GEYYxqxKru5upP2/BLoB5T3p1TVS4MYl/Hj8PFC9ucVcqdVLxljgiiQO6ln4Uz40xNnNNfdODO+mRBZn+G2P1gDtTEmiAJJEO1V9W9AsaouUdU7gdFBjsv4sT4zlwiB/l1ahToUY0wjFshQG8Xu3wMiMgHYDyQELyRTlXWZefSJb0mzmED++YwxpmYCuYL4nTsO089x5ol+DXggkI2LyHgR2SoiO0TkkUrK3Cgim0Rko4i8XeG9ViKyT0ReDGR/ZwNVZb17B7UxxgSTv/sgklR1tap+7C7KA8YFumERiQSmA1cAmcAqEZmnqpu8yvQGHsUZUvyoiMRV2MxvgSWB7vNskJFzktwTxQyyAfqMMUHm7wriVRHZLiJPiEi/Gmx7JLBDVdNVtQiYDVxXoczdwHRVPQqgqofL3hCR4UA88O8a7LvRWpfpjuBqVxDGmCDzdx/EUBHpC0wC3hORIuAdYLbXPNX+dAUyvF5nAqMqlOkDICJfAJHANFX9VEQigD8D3wcuq2wHInIPcA9AfHw8KSkpAYQVnvLz8wOKf/6WU0RHwMGtaziyPXwG6Qs0/nBl8YeWxR+e/LZyqupWnK6tvxGRwTjJ4r8iclBVL6pi277OXhXnuI4CegPJOA3fS0VkAHArsEBVM/yNVKqqM4AZAElJSZqcnFxFSOErJSWFQOJ/actyBiZ4uPzSqj7++hVo/OHK4g8tiz88BdQNxv1FH4dT5dMcyApgtUygm9frBJweUBXLrFDVYmCXiGzFSRgXAJeIyL1ACyBGRPLd6U7PWiWlHr7el8dNI7pVXdgYY2rJby8mEblERF7COZE/DCwD+qrqdwLY9iqgt4j0FJEYnKuPeRXKfIDb8C0iHXCqnNJV9RZV7a6qiTg9p/5xticHgB1Z+ZwsLrUZ5Iwx9cJfL6YMYC9O4/JvVPVQdTasqiUiMhVYiNO+MFNVN4rIE8BqVZ3nvneliGwCSoGHVTW7hsfS6JXfQW0N1MaYeuCviuniABujK6WqC4AFFZY95vVcgQfdR2Xb+Dvw99rE0Visy8ylZWwUie2bV13YGGNqqdIqptomB1P31mfmMiihNRER4dN7yRjTeAVyJ7UJA4XFpWw5cNyql4wx9cYSRAOx6cAxSjzKIEsQxph6UmWCEJGn3DGRokVksYgcEZFb6yM48431Ge4d1NaDyRhTTwK5grhSVY8BV+N0d+2D0+XV1KO0zDziWjahU6vYqgsbY0wdCCRBRLt/r1FYclYAABv3SURBVALeUdWcIMZjKpGWmcughDb4u7PcGGPqUiAJ4iMR2QIkAYvdKUgLgxuW8XassJj0rAIG2xSjxph6VGWCcO9gvgBIcofEKODMUVlNEH2daVOMGmPqXyCN1N8DSlS1VER+DbwFdAl6ZKZcmjvE9yC7gjDG1KNAqpj+V1WPi8jFwLeAN4CXgxuW8bY+I4/E9s1o0ywm1KEYY84igSSIUvfvBOBlVf0QsDNVPSproDbGmPoUSILYJyJ/BW4EFohIkwDXM3Xg8PFCDuQVWvWSMabeBXKivxFn1NXxqpoLtMPug6g3ZSO4DrEGamNMPQukF9MJYCfwLXf47jhVtXmi60laZi6REUL/LnYFYYypX4H0YrofmIUzo1wc8JaI3BfswIwjLTOP3nEtaBoTGepQjDFnmUCmHP0BMEpVCwBE5I/AcuCFYAZmQFVZn5nL+P6dQh2KMeYsFEgbhPBNTybc5zbeQz3Ym3OC3BPF1oPJGBMSgVxBvA58JSJz3dffAf4WvJBMmTT3DmrrwWSMCYUqE4SqPiMiKcDFOFcOd6jq2mAHZpwhvptERdC3U8tQh2KMOQv5rWISkQgR2aCqa1T1eVX9S3WSg4iMF5GtIrJDRB6ppMyNIrJJRDaKyNvusiEistxdtl5EbqreYTUOaZm59O/SiuhIu+3EGFP//F5BqKpHRNJEpLuq7q3OhkUkEpgOXIEzj8QqEZmnqpu8yvQGHgUuUtWjIhLnvnUCuE1Vt4tIFyBVRBa692GcFUpKPWzYd4ybRnQLdSjGmLNUIG0QnYGNIrISZyRXAFT12irWGwnsUNV0ABGZjTMK7CavMncD01X1qLvNw+7fbV772S8ih4GOwFmTILYfzudkcandIGeMCZlAEsRvarjtrkCG1+tMYFSFMn0AROQLIBKYpqqfehcQkZE4Yz/trLgDEbkHuAcgPj6elJSUGoYaevn5+afFvySzGIDC/VtJydseoqgCVzH+hsbiDy2LPzxVmiBE5FwgXlWXVFg+BtgXwLZ9dYVVH/vvDSQDCcBSERlQVpUkIp2BN4HbVdVzxsZUZwAzAJKSkjQ5OTmAsMJTSkoK3vH/e+7XtIrdz43fHkdERPj3Kq4Yf0Nj8YeWxR+e/LV+Pgcc97H8hPteVTIB7wr0BGC/jzIfqmqxqu4CtuIkDESkFTAf+LWqrghgf41KWoYzgmtDSA7GmMbJX4JIVNX1FReq6mogMYBtrwJ6i0hPEYkBJgHzKpT5ABgHICIdcKqc0t3yc4F/qOq7AeyrUSksLmXrweN2/4MxJqT8JYhYP+81rWrDqloCTMUZCXYz8E9V3SgiT4hIWQP3QiBbRDYBnwEPq2o2zgiyY4ApIrLOfQwJ4HgahU0HjlHiUZti1BgTUv4aqVeJyN2q+qr3QhH5AZAayMZVdQGwoMKyx7yeK/Cg+/Au8xbO1KZnpbQMp7PWYBtiwxgTQv4SxM+AuSJyC98khCScHkUTgx3Y2Wx9Zh5xLZvQqbW/izhjjAmuShOEqh4CLhSRccAAd/F8Vf1vvUR2FkvLzLXqJWNMyAUyFtNnOO0Dph7knSwmPauA64d2DXUoxpiznA3yE2Y27CsbwdWuIIwxoWUJIsykZToN1NbF1RgTapYgwkxaRi6J7ZvRpllMqEMxxpzlLEGEmfWZeVa9ZIwJC5YgwsjhY4UcyCu0HkzGmLBgCSKMlE0xOtjaH4wxYcASRBhZn5lLZITQv4slCGNM6FmCCCPrMnLpE9+SpjGRoQ7FGGMsQYQLVeXrfXlWvWSMCRuWIMJE1kkl90SxNVAbY8KGJYgwkZ7rTJhnN8gZY8KFJYgwsSuvlCZREfSJbxnqUIwxBrAEETZ2HfMwoGtroiPtn8QYEx7sbBQGSko97M7zWPWSMSasWIIIA9sP51PksRnkjDHhJagJQkTGi8hWEdkhIo9UUuZGEdkkIhtF5G2v5beLyHb3cXsw4wy19e4IrtaDyRgTTqqcMKimRCQSmA5cAWTizHE9T1U3eZXpDTwKXKSqR0Ukzl3eDngcZ4pTBVLddY8GK95QWpeRR7MoSGzfLNShGGNMuWBeQYwEdqhquqoWAbOB6yqUuRuYXnbiV9XD7vJvAYtUNcd9bxEwPoixhtSK9CO0iBHW7M0NdSjGGFMuaFcQQFcgw+t1JjCqQpk+ACLyBRAJTFPVTytZ94w5OEXkHuAegPj4eFJSUuoq9nqz6mAxu44UAcrkv37JL0bEcm7bhjfURn5+foP8/MtY/KFl8YenYCYI8bFMfey/N5AMJABLRWRAgOuiqjOAGQBJSUmanJxci3DrX1GJh0ee+q/7SihVONWmB8nJ54Y0rppISUmhoX3+3iz+0LL4w1Mwq5gygW5erxOA/T7KfKiqxaq6C9iKkzACWbfBe/KTLRw8doroSCECiI6KYHSv9qEOyxhjgOAmiFVAbxHpKSIxwCRgXoUyHwDjAESkA06VUzqwELhSRNqKSFvgSndZo/HvjQeZ+cUuplyYyOx7LuD63tHMums0w3u0DXVoxhgDBLGKSVVLRGQqzok9EpipqhtF5AlgtarO45tEsAkoBR5W1WwAEfktTpIBeEJVc4IVa33LyDnBQ++mMbBrax696jyaREVy/JwYSw7GmLASzDYIVHUBsKDCsse8nivwoPuouO5MYGYw4wuFohIPU99ZiypMv3kYTaIaXoO0MebsENQEYc70x0+3kJaRy8u3DKO73fdgjAljNtRGPfr3xoP8bdkubr+gB98e2DnU4RhjjF+WIOpJWbvDgK6t+J8J54c6HGOMqZJVMdWDohIP91m7g6mB4uJiMjMzKSws9FuudevWbN68uZ6iqnsWf/DFxsaSkJBAdHR0wOtYgqgHT326hXUZubx0yzB6tG8e6nBMA5KZmUnLli1JTExExNf9o47jx4/TsmXDnWzK4g8uVSU7O5vMzEx69uwZ8HpWxRRkizYd4rVlu7jtgh5cZe0OppoKCwtp37693+RgTFVEhPbt21d5JVqRJYggyjzq1e5wlbU7mJqx5GDqQk2+R5YggqSoxMPUt9fi8SjTbx5GbLS1OxhjGhZLEEHyp4VOu8OTNwyydgfToB08eJBJkyZxzjnn0K9fP6666iq2bdsW1H3u3r2bhIQEPB7PacuHDBnCypUrK13v73//O1OnTgXglVde4R//+IfPbQ8YMKDK/b/9dvn8ZaxevZqf/vSn1TmESs2cOZOBAwcyaNAgBgwYwIcfflgn2w0Ga6QOgv9sOsSrS3fx/dE9mDDI2h1Mw6WqTJw4kdtvv53Zs2cDsG7dOg4dOkSfPn3Ky5WWlhIZWXdXyYmJiXTr1o2lS5cyduxYALZs2cLx48cZOXJkQNv40Y9+VOP9lyWIm2++GYCkpCSSkpJqvL0ymZmZ/P73v2fNmjW0bt2a/Px8srKyarXNuv7svdkVRB3bl3uSn7+bRv8urfiV3e9gQiB1z1Gmf7aD1D21n4Dxs88+Izo6+rST7ZAhQ7jkkktISUlh3Lhx3HzzzQwcOBCAZ555hgEDBjBgwACee+45AAoKCpgwYQKDBw9mwIABzJkzB4BHHnmEfv36MWjQIH71q1+dse/JkyeXJyWA2bNnM3nyZAA++ugjRo0axdChQ7n88ss5dOjQGetPmzaNp59+2vlMUlMZPHgwF1xwAdOnTy8vs3v3bi655BKGDRvGsGHD+PLLL8tjW7p0KUOGDOHZZ58lJSWFq6++GoCcnBy+853vMGjQIEaPHs369evL93fnnXeSnJxMr169eP7558+I6fDhw7Rs2ZIWLVoA0KJFi/JeRTt27ODyyy9n8ODBDBs2jJ07d6KqPPzwwwwYMICBAweWf3a+Pvu33nqLkSNHMmTIEH74wx9SWlrq7582IHYFUYeKSz1MfXsNpdbuYILgNx9tZNP+Yz7fK/sVebywmC0Hj+NRiBA4r1NLWsZW3u+9X5dWPH5N/0rf37BhA8OHD6/0/ZUrV7JhwwZ69uxJamoqr7/+Ol999RWqyqhRoxg7dizp6el06dKF+fPnA5CXl0dOTg5z585ly5YtiAgZGRlnbPvGG29k6NChvPDCC0RFRTFnzhzeffddAC6++GJWrFiBiPDaa6/x1FNP8ec//7nSOO+44w5eeOEFxo4dy8MPP1y+PC4ujkWLFhEbG8v27duZPHkyq1ev5sknn+Tpp5/m448/BjhtMqDHH3+coUOH8sEHH/Df//6X2267jaVLlwLOVc5nn33G8ePH6du3Lz/+8Y9Pu+9g8ODBxMfH07NnTy677DKuv/56rrnmGgBuueUWHnnkESZOnEhhYSEej4f333+fdevWkZaWxpEjRxgxYgRjxow547PfvHkzc+bM4YsvviA6Opp7772XWbNmcdttt1X6mQTCEkQd+tPCrazdm8uLNw8lsYO1O5j6d6ywBI87tZZHndf+EkRtjRw5svwX8LJly5g4cSLNmzvf/euvv56lS5cyfvx4HnroIX75y19y9dVXc8kll1BSUkJsbCx33XUXEyZMKK9G8tapUyf69+/P4sWLiY+PJzo6urztIDMzk5tuuokDBw5QVFTkt29/Xl4eubm55fv4/ve/zyeffAI4NyJOnTqVdevWERkZGVDbyrJly/jXv/4FwKWXXkp2djZ5eXkATJgwgSZNmtCkSRPi4uI4dOgQCQkJ5etGRkby6aefsmrVKhYvXswDDzxAamoqP//5z9m3bx8TJ04EnJvayvY1efJkIiMjiY+PZ+zYsaxatYpWrVqd9tkvXryY1NRURowYAcDJkyeJi4ur8liqYgmijizefIgZn6dz6+juXD2oS6jDMY2Qv1/6ZTdqpe45yi2vraC4xEN0VAR/mTS0VsPI9+/fn/fee6/S98uSATjtFb706dOH1NRUFixYwKOPPsqVV17JY489xsqVK1m8eDGzZ8/mL3/5C0uWLDlj3bJqpvj4+PLqJYD77ruPBx98kGuvvZaUlBSmTZtWaYyqWmkXz2effZb4+HjS0tLweDzlJ2Z/fB1n2fabNGlSviwyMpKSkhKfZUeOHMnIkSO54ooruOOOO3jwwTMGtK50X2Uqfva33347f/jDH6qMvzqsDYLa19ku3HiQn7y9hsT2zfj1hH51HJ0xgRveoy2z7hrNg1f2rZMJqC699FJOnTrFq6++Wr5s1apVPk/mY8aM4YMPPuDEiRMUFBQwd+5cLrnkEvbv30+zZs249dZbeeihh1izZg35+fnk5eVx1VVX8dxzz5XX41d0ww03sGDBAubMmcOkSZPKl+fl5dG1qzNN/RtvvOH3GNq0aUPr1q1ZtmwZALNmzTptO507dyYiIoI333yzvN6+ZcuWHD9+3Of2xowZU76NlJQUOnToQKtWrfzGUGb//v2sWbOm/PW6devo0aMHrVq1IiEhgQ8++ACAU6dOceLECcaMGcOcOXMoLS0lKyuLzz//3Gcj/WWXXcZ7773H4cOHAaedZM+ePQHF5M9ZfwWxdHsWU2auolSVSBEu7t2e9s2bVL2iK7vgFJ9vO4ICB/IK2bj/mE38Y0JqeI+2dfYdFBHmzp3Lz372M5588kliY2NJTEzkueeeY9++faeVHTZsGFOmTCk/gd11110MHTqUhQsX8vDDDxMREUF0dDQvv/wyx48f57rrrqOwsBBVrfSXb5s2bRg9ejSHDh06rRpp2rRpfO9736Nr166MHj2aXbt2+T2O119/nTvvvJNmzZrxrW99q3z5vffeyw033MC7777LuHHjyn+VDxo0iKioKAYPHsyUKVMYOnToafu+4447GDRoEM2aNasyQXkrLi7moYceYv/+/cTGxtKxY0deeeUVAN58801++MMf8thjjxEdHc27777LxIkTWb58OYMHD0ZEeOqpp+jUqRNbtmw5bbv9+vXjd7/7HVdeeSUej4fo6GimT59Ojx49Ao7NF/F3CdOQJCUl6erVq6u93tMLt/LiZzvKX7duGkWrpoHX2R47WUzeSecyMlLgwSv78pNx51Y7joY+6bnFHxybN2/m/POr7g0X7mMBVcXirx++vk8ikqqqPvvwnvVXEOPOi+O1ZenldbYzp4ys1q+vinW+o3u1D2K0xhhTf876BFFWZ7siPZvRvdpX+9K8tusbY0y4CmqCEJHxwF+ASOA1VX2ywvtTgD8BZZWZL6rqa+57TwETcBrSFwH3a5Dqw2pbZ1uXdb7GVOSvF44xgarJ6TNovZhEJBKYDnwb6AdMFhFfXXzmqOoQ91GWHC4ELgIGAQOAEcCZHaWNaeRiY2PJzs6u0X9uY8qUzQcRSDdeb8G8ghgJ7FDVdAARmQ1cB2wKYF0FYoEYQIBo4Mx76Y1p5BISEsjMzKxyvJ7CwsJq/+cPJxZ/8JXNKFcdwUwQXQHv++czgVE+yt0gImOAbcADqpqhqstF5DPgAE6CeFFVz5jPT0TuAe4BiI+PP+12+IYmPz/f4g+hxhB/2fg+DZHFXz+qe29EMBOEr0rTitfJHwHvqOopEfkR8AZwqYicC5wPlKW7RSIyRlU/P21jqjOAGeB0cw3HboqBCtduloGy+EPL4g+thh5/ZYJ5J3Um0M3rdQKw37uAqmar6in35atA2ahgE4EVqpqvqvnAJ8DoIMZqjDGmgmAmiFVAbxHpKSIxwCRgnncBEfGeLOFaoKwaaS8wVkSiRCQap4H6jComY4wxwRO0KiZVLRGRqcBCnG6uM1V1o4g8AaxW1XnAT0XkWqAEyAGmuKu/B1wKfI1TLfWpqn7kb3+pqalHRKT2g4+ETgfgSKiDqAWLP7Qs/tBqyPFXOh5Hoxlqo6ETkdWV3e7eEFj8oWXxh1ZDj78yNpqrMcYYnyxBGGOM8ckSRPiYEeoAasniDy2LP7Qaevw+WRuEMcYYn+wKwhhjjE+WIIwxxvhkCcIYY4xPZ/2EQeFORCKA3wKtcG4wDHwC3DAgIucD9+PcSLRYVV8OcUhVEpHmwEtAEZCiqrOqWCXsNMTP3Vsj+N73A6YB2Tif/3uhjahm7AoiiERkpogcFpENFZaPF5GtIrJDRB6pYjPX4YyMW4wzvlW9qYv4VXWzqv4IuBEI2Y1E1TyW64H3VPVunCFgwkJ1jiFcPndv1fw3CNn3vjLVjP/bwAuq+mPgtnoPtq6oqj2C9ADGAMOADV7LIoGdQC+c+S7ScCZUGgh8XOERBzwC/NBd972GFr+7zrXAl8DNDeTf4lFgiFvm7VB/j2pyDOHyudfi3yBk3/s6ij8OZ8K0PwFfhDr2mj6siimIVPVzEUmssNjnREqq+gfg6orbEJFMnKoOgNLgRXumuojf3c48YJ6IzAfeDl7ElavOseD8Yk0A1hFGV9nVPIZN4fC5e6tm/BmE6HtfmRr8f/iJO7Pm+/UaaB2yBFH/Ap1Iqcz7wAsicgnwuZ9y9aVa8YtIMk6VTRNgQVAjq77KjuV54EURmYAzZ0k483kMYf65e6vs3+AvhNf3vjKVff6JwP8AzXGuIhokSxD1L5CJlL55Q/UE8IPghVNt1Y0/BUgJVjC15PNYVLUAuKO+g6mhyo4hhfD93L1VFn+4fe8rU1n8u3Fnu2zIwuby+SxS5URKYa6hx++tMRxLQz8Giz+MWYKof1VOpBTmGnr83hrDsTT0Y7D4w5gliCASkXeA5UBfEckUkR+oaglQNpHSZuCfqroxlHFWpqHH760xHEtDPwaLv+GxwfqMMcb4ZFcQxhhjfLIEYYwxxidLEMYYY3yyBGGMMcYnSxDGGGN8sgRhjDHGJ0sQpk6JSKmIrBORDSLykYi0CcI+kkXk42qu00VEqj0mv4i0EZF7a7udSrad4g4TnSYiX4hI37rYbm2JyBQR6VLH2+wjIgvcIbE3i8g/RSS+Lvdh6p4lCFPXTqrqEFUdAOQAPwl1QCISpar7VfW7NVi9DVCeIGqxncrcoqqDgTeoxqBuIhLMcdSmANVKEP7iEZFYYD7wsqqeq6rnAy8DHWsTpAk+SxAmmJbjjHYJgIg8LCKrRGS9iPzGa/n/isgWEVkkIu+IyEPu8hQRSXKfdxCR3RV3ICIjReRLEVnr/u3rLp8iIu+KyEfAv0UksWyiFxF5zb3KWSciWSLyuIi0EJHFIrJGRL4WkevcXTwJnOOW/VOF7cSKyOtu+bUiMs5r3++LyKcisl1Engrgs/ocONdd/zH3c9ogIjNERLw+j/8TkSXA/SJyjYh85e77P2W/yEVkmoi8ISL/FpHdInK9iDzlxvmpiES75YaLyBIRSRWRhSLSWUS+izPB0Cz3mJv6KucrHj/HdjOwXFXLR8ZV1c9UdYOfdUw4CPWEFPZoXA8g3/0bCbwLjHdfXwnMwBn9MgJnQqExOCejdUBToCWwHXjIXScFSHKfdwB2u8+TgY/d562AKPf55cC/3OdTcAZSa+e+TsRrohd3WQ9gi/s3Cmjlta8dbqynref9Gvg58Lr7/DxgLxDr7jsdaO2+3gN08/FZeR/fw8Ac93k7rzJvAtd4lX/J6722fDMawl3An93n04BlQDQwGDgBfNt9by7wHfe9L4GO7vKbgJk+4qqqnHc81wJP+DjOZ4D7Q/3dtEf1Hzbct6lrTUVkHc6JNBVY5C6/0n2sdV+3AHrjJIUPVfUkgPuLvzpaA2+ISG+cYcejvd5bpKo5vlZyqz3eBaaq6h73V/X/icgYwINz5VNVHfnFwAsAqrpFRPYAfdz3FqtqnruvTThJKMPHNmaJyElgN3Cfu2yciPwCaAa0AzbyzbwUc7zWTQDmuL/oY4BdXu99oqrFIvI1TrL+1F3+Nc6/TV9gALDIvUCJBA74iK+qcuXxqDtBkY9tmAbKEoSpaydVdYiItMa5SvgJzgQ8AvxBVf/qXVhEHvCzrRK+qQaNraTMb4HPVHWiOJO0pHi9V+Bn268A76vqf9zXt+DUiQ93T6y7/eyzjK+5AMqc8npeSuX/125R1dXlG3QS10s4v+AzRGRahTi8j+kF4BlVnSfOBEHTKu5fVT0iUqzuT3mc5Bflxr5RVS/wcwwEUM7fZ1xmIzA2gHImzFgbhAkK99fzT4GH3F/nC4E7RaQFgIh0FZE4nKqQa9z6/BbABK/N7AaGu88raxhuDexzn08JJDYR+QnQUlWfrLCdw25yGIfzix/gOM5Vji+f4yQWRKQP0B3YGkgMfpQlgyPu5+GvQdz72G+v5n62Ah1F5AIAEYkWkf7ue97H7K9coN4GLhRnhj7c7YwXkYHV3I6pZ5YgTNCo6lqcSdwnqeq/cU4Uy91qj/dwTtKrcKol0nCmV10N5LmbeBr4sYh8idMu4MtTwB9E5Auc6o9APAQM9Gqo/hEwC0gSkdU4J/0t7jFkA1+4DcYVexm9BES6xzMHmKKqp6gFVc0FXsWpCvoAZ76BykwD3hWRpcCRau6nCCf5/FFE0nDagS503/478IpbVRjpp9xpRORaEXnCx75O4sxXfp/baL8JJ5kfrk7Mpv7ZcN8m5ESkharmi0gznF/l96jqmlDHZczZztogTDiYISL9cKpX3rDkYEx4sCsIY4wxPlkbhDHGGJ8sQRhjjPHJEoQxxhifLEEYY4zxyRKEMcYYnyxBGGOM8en/AT79ZIMmPNCBAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(C_range, cross_validation_scores,label=\"Cross Validation Score\",marker='.')\n",
    "plt.legend()\n",
    "plt.xscale(\"log\")\n",
    "plt.xlabel('Regularization Parameter: C')\n",
    "plt.ylabel('Cross Validation Score')\n",
    "plt.grid()\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Based on the chart above, with c = 10^6 we achive the highest cross validation score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The test accuracy is: 0.718\n"
     ]
    }
   ],
   "source": [
    "#With our model tuned, lets test it out!\n",
    "LR_model = LogisticRegression(C=10**6, random_state=1)\n",
    "LR_model.fit(X_train, y_train)\n",
    "\n",
    "print(f'The test accuracy is: {LR_model.score(X_test,y_test):0.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 5.2 What does your confusion matrix look like for your best model on the test set?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 75  55]\n",
      " [ 22 121]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "y_pred = LR_model.predict(X_test)\n",
    "\n",
    "print(confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "|     |  Class1 Predicted  |   Class2 Predicted  | \n",
    "| --------- |:---:|:---:|\n",
    "|Class1 Actual |75 | 55 | \n",
    "|Class2 Actual |22 |121 | "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- 75 times out of 97 times, our model predicted correctly a True Positve (77% Precision)!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.77      0.58      0.66       130\n",
      "         1.0       0.69      0.85      0.76       143\n",
      "\n",
      "    accuracy                           0.72       273\n",
      "   macro avg       0.73      0.71      0.71       273\n",
      "weighted avg       0.73      0.72      0.71       273\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "report_initial = classification_report(y_test, y_pred)\n",
    "print(report_initial)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 6. Create one new feature of your choice: "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 6.1 Explain your new feature and why you consider it will improve accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Knowing that summer is peak travel season, I wonder if knowing the summer rush would increase the accuracy of predicting a postive review (Revewier_Score = 1)\n",
    "- There is a large increase in the number of students and others traveling around europe. Do they cause a ruckus and or is the general volume of people lead to more hotels being filled, more demmand on staff, more demmand on cleaning?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 6.2 Run the model from question 5 again. You will have to re-optimize your hyperparameters. Has the accuracy score of your best model improved on the test set after adding the new feature you created?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "#First up, lets refresh the data\n",
    "df_train = pd.read_csv(\"data/train_dataframe.csv\")\n",
    "df_test = pd.read_csv(\"data/test_dataframe.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using the LogisticRegression with parameters (solver='lbfgs', max_iter=10000)\n",
    "- The original train accuracy of the model is: 0.811\n",
    "- The original test accuracy of the model is: 0.785"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Additional_Number_of_Scoring</th>\n",
       "      <th>Average_Score</th>\n",
       "      <th>Review_Total_Negative_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews</th>\n",
       "      <th>Review_Total_Positive_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>lat</th>\n",
       "      <th>lng</th>\n",
       "      <th>Review_Month</th>\n",
       "      <th>...</th>\n",
       "      <th>p_working</th>\n",
       "      <th>p_world</th>\n",
       "      <th>p_worth</th>\n",
       "      <th>p_wouldn</th>\n",
       "      <th>p_year</th>\n",
       "      <th>p_years</th>\n",
       "      <th>p_yes</th>\n",
       "      <th>p_young</th>\n",
       "      <th>p_yummy</th>\n",
       "      <th>Reviewer_Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>220.0</td>\n",
       "      <td>9.1</td>\n",
       "      <td>20.0</td>\n",
       "      <td>902.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>275.0</td>\n",
       "      <td>51.494308</td>\n",
       "      <td>-0.175558</td>\n",
       "      <td>11.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1190.0</td>\n",
       "      <td>7.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5180.0</td>\n",
       "      <td>23.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>481.0</td>\n",
       "      <td>51.514879</td>\n",
       "      <td>-0.160650</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.425849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>299.0</td>\n",
       "      <td>8.3</td>\n",
       "      <td>81.0</td>\n",
       "      <td>1361.0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>672.0</td>\n",
       "      <td>51.521009</td>\n",
       "      <td>-0.123097</td>\n",
       "      <td>10.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>87.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>355.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>412.0</td>\n",
       "      <td>51.499749</td>\n",
       "      <td>-0.161524</td>\n",
       "      <td>6.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>317.0</td>\n",
       "      <td>7.6</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1458.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>499.0</td>\n",
       "      <td>51.516114</td>\n",
       "      <td>-0.174952</td>\n",
       "      <td>3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2587 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Additional_Number_of_Scoring  Average_Score  \\\n",
       "0                         220.0            9.1   \n",
       "1                        1190.0            7.5   \n",
       "2                         299.0            8.3   \n",
       "3                          87.0            9.0   \n",
       "4                         317.0            7.6   \n",
       "\n",
       "   Review_Total_Negative_Word_Counts  Total_Number_of_Reviews  \\\n",
       "0                               20.0                    902.0   \n",
       "1                                5.0                   5180.0   \n",
       "2                               81.0                   1361.0   \n",
       "3                               17.0                    355.0   \n",
       "4                               14.0                   1458.0   \n",
       "\n",
       "   Review_Total_Positive_Word_Counts  \\\n",
       "0                               21.0   \n",
       "1                               23.0   \n",
       "2                               27.0   \n",
       "3                               13.0   \n",
       "4                                0.0   \n",
       "\n",
       "   Total_Number_of_Reviews_Reviewer_Has_Given  days_since_review        lat  \\\n",
       "0                                         1.0              275.0  51.494308   \n",
       "1                                         6.0              481.0  51.514879   \n",
       "2                                         4.0              672.0  51.521009   \n",
       "3                                         7.0              412.0  51.499749   \n",
       "4                                         1.0              499.0  51.516114   \n",
       "\n",
       "        lng  Review_Month  ...  p_working  p_world  p_worth  p_wouldn  p_year  \\\n",
       "0 -0.175558          11.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "1 -0.160650           4.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "2 -0.123097          10.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "3 -0.161524           6.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "4 -0.174952           3.0  ...        0.0      0.0      0.0       0.0     0.0   \n",
       "\n",
       "    p_years  p_yes  p_young  p_yummy  Reviewer_Score  \n",
       "0  0.000000    0.0      0.0      0.0             1.0  \n",
       "1  0.425849    0.0      0.0      0.0             1.0  \n",
       "2  0.000000    0.0      0.0      0.0             0.0  \n",
       "3  0.000000    0.0      0.0      0.0             1.0  \n",
       "4  0.000000    0.0      0.0      0.0             0.0  \n",
       "\n",
       "[5 rows x 2587 columns]"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Months 5, 6, 7, 8 will considered to be summer months\n",
    "def is_summer(review_month):\n",
    "    if review_month < 5:\n",
    "        return (0)\n",
    "    elif review_month > 8:\n",
    "        return (0)\n",
    "    else:\n",
    "        return 1\n",
    "    \n",
    "# df_test = df_train.apply(lambda x: func(x['col1'],x['col2']),axis=1)\n",
    "df_new = df_train.apply(lambda x: is_summer(x['Review_Month']), axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "# pd.concat([df1, df3], sort=False)\n",
    "df_train_new = pd.concat([df_train, df_new], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train_new = df_train_new.rename(columns = {0: \"is_summer\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Additional_Number_of_Scoring</th>\n",
       "      <th>Average_Score</th>\n",
       "      <th>Review_Total_Negative_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews</th>\n",
       "      <th>Review_Total_Positive_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>lat</th>\n",
       "      <th>lng</th>\n",
       "      <th>Review_Month</th>\n",
       "      <th>...</th>\n",
       "      <th>p_world</th>\n",
       "      <th>p_worth</th>\n",
       "      <th>p_wouldn</th>\n",
       "      <th>p_year</th>\n",
       "      <th>p_years</th>\n",
       "      <th>p_yes</th>\n",
       "      <th>p_young</th>\n",
       "      <th>p_yummy</th>\n",
       "      <th>Reviewer_Score</th>\n",
       "      <th>is_summer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>220.0</td>\n",
       "      <td>9.1</td>\n",
       "      <td>20.0</td>\n",
       "      <td>902.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>275.0</td>\n",
       "      <td>51.494308</td>\n",
       "      <td>-0.175558</td>\n",
       "      <td>11.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1190.0</td>\n",
       "      <td>7.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5180.0</td>\n",
       "      <td>23.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>481.0</td>\n",
       "      <td>51.514879</td>\n",
       "      <td>-0.160650</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.425849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>299.0</td>\n",
       "      <td>8.3</td>\n",
       "      <td>81.0</td>\n",
       "      <td>1361.0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>672.0</td>\n",
       "      <td>51.521009</td>\n",
       "      <td>-0.123097</td>\n",
       "      <td>10.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>87.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>355.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>412.0</td>\n",
       "      <td>51.499749</td>\n",
       "      <td>-0.161524</td>\n",
       "      <td>6.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>317.0</td>\n",
       "      <td>7.6</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1458.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>499.0</td>\n",
       "      <td>51.516114</td>\n",
       "      <td>-0.174952</td>\n",
       "      <td>3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2588 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Additional_Number_of_Scoring  Average_Score  \\\n",
       "0                         220.0            9.1   \n",
       "1                        1190.0            7.5   \n",
       "2                         299.0            8.3   \n",
       "3                          87.0            9.0   \n",
       "4                         317.0            7.6   \n",
       "\n",
       "   Review_Total_Negative_Word_Counts  Total_Number_of_Reviews  \\\n",
       "0                               20.0                    902.0   \n",
       "1                                5.0                   5180.0   \n",
       "2                               81.0                   1361.0   \n",
       "3                               17.0                    355.0   \n",
       "4                               14.0                   1458.0   \n",
       "\n",
       "   Review_Total_Positive_Word_Counts  \\\n",
       "0                               21.0   \n",
       "1                               23.0   \n",
       "2                               27.0   \n",
       "3                               13.0   \n",
       "4                                0.0   \n",
       "\n",
       "   Total_Number_of_Reviews_Reviewer_Has_Given  days_since_review        lat  \\\n",
       "0                                         1.0              275.0  51.494308   \n",
       "1                                         6.0              481.0  51.514879   \n",
       "2                                         4.0              672.0  51.521009   \n",
       "3                                         7.0              412.0  51.499749   \n",
       "4                                         1.0              499.0  51.516114   \n",
       "\n",
       "        lng  Review_Month  ...  p_world  p_worth  p_wouldn  p_year   p_years  \\\n",
       "0 -0.175558          11.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "1 -0.160650           4.0  ...      0.0      0.0       0.0     0.0  0.425849   \n",
       "2 -0.123097          10.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "3 -0.161524           6.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "4 -0.174952           3.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "\n",
       "   p_yes  p_young  p_yummy  Reviewer_Score  is_summer  \n",
       "0    0.0      0.0      0.0             1.0          0  \n",
       "1    0.0      0.0      0.0             1.0          0  \n",
       "2    0.0      0.0      0.0             0.0          0  \n",
       "3    0.0      0.0      0.0             1.0          1  \n",
       "4    0.0      0.0      0.0             0.0          0  \n",
       "\n",
       "[5 rows x 2588 columns]"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train_new.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting up our variables (With the new df)\n",
    "y_train = df_train_new['Reviewer_Score']\n",
    "X_train = df_train_new.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "#Sub-sampling our data for speedy test and checks\n",
    "X_train = X_train.sample(frac=.1, random_state=42)\n",
    "y_train = y_train.sample(frac=.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting out 20% for test set\n",
    "X_remainder, X_test, y_remainder, y_test = train_test_split(X_train, y_train, test_size = 0.2,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.75342466 0.73059361 0.74311927 0.69724771 0.72477064]\n"
     ]
    }
   ],
   "source": [
    "# Takes about 2 min to run\n",
    "# 1. Instanitate model\n",
    "my_logreg = LogisticRegression(random_state=1, max_iter=10000)\n",
    "\n",
    "# 2. Fit model on 5 folds.\n",
    "# The variable \"scores\" will hold 5 accuracy scores, \n",
    "# each from a different train and validation split\n",
    "scores = cross_val_score(my_logreg, X_remainder, y_remainder, cv = 5)\n",
    "print(scores)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Scores from last model: 0.72146119 0.71689498 0.75229358 0.68348624 0.70183486\n",
    "- Score from this model: 0.71232877 0.72146119 0.75688073 0.68348624 0.71559633"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Finished cross validation with c = 100000000000.0..\r"
     ]
    }
   ],
   "source": [
    "#Store the results\n",
    "cross_validation_scores = []\n",
    "\n",
    "C_range = np.array([.00000001,.0000001,.000001,.00001,.0001,.001,.1,\\\n",
    "                1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000,10000000000,100000000000])\n",
    "\n",
    "#Do some cross validation\n",
    "for c in C_range:\n",
    "    LR_model = LogisticRegression(C=c, random_state=1)\n",
    "    \n",
    "    # the cross validation score (mean of scores from all folds)\n",
    "    cv_score = np.mean(cross_val_score(LR_model, X_remainder, y_remainder, cv = 5))\n",
    "    \n",
    "    cross_validation_scores.append(cv_score)\n",
    "    print(f'Finished cross validation with c = {c}..', end='\\r')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(C_range, cross_validation_scores,label=\"Cross Validation Score\",marker='.')\n",
    "plt.legend()\n",
    "plt.xscale(\"log\")\n",
    "plt.xlabel('Regularization Parameter: C')\n",
    "plt.ylabel('Cross Validation Score')\n",
    "plt.grid()\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- C = 10000 looks to yield the highest cross validation score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-84-13a300106785>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# With our model tuned, lets test it out!\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mLR_model\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mLogisticRegression\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mC\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10000\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10000\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mLR_model\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf'The test accuracy is: {LR_model.score(X_test,y_test):0.3f}'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m   1599\u001b[0m                       \u001b[0mpenalty\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mpenalty\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmax_squared_sum\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmax_squared_sum\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1600\u001b[0m                       sample_weight=sample_weight)\n\u001b[0;32m-> 1601\u001b[0;31m             for class_, warm_start_coef_ in zip(classes_, warm_start_coef))\n\u001b[0m\u001b[1;32m   1602\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1603\u001b[0m         \u001b[0mfold_coefs_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_iter_\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mfold_coefs_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, iterable)\u001b[0m\n\u001b[1;32m   1002\u001b[0m             \u001b[0;31m# remaining jobs.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1003\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1004\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdispatch_one_batch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1005\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_original_iterator\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1006\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36mdispatch_one_batch\u001b[0;34m(self, iterator)\u001b[0m\n\u001b[1;32m    833\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    834\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 835\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_dispatch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtasks\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    836\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    837\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m_dispatch\u001b[0;34m(self, batch)\u001b[0m\n\u001b[1;32m    752\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    753\u001b[0m             \u001b[0mjob_idx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_jobs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 754\u001b[0;31m             \u001b[0mjob\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mapply_async\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    755\u001b[0m             \u001b[0;31m# A job can complete so quickly than its callback is\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    756\u001b[0m             \u001b[0;31m# called before we get here, causing self._jobs to\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py\u001b[0m in \u001b[0;36mapply_async\u001b[0;34m(self, func, callback)\u001b[0m\n\u001b[1;32m    207\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mapply_async\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    208\u001b[0m         \u001b[0;34m\"\"\"Schedule a func to be run\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 209\u001b[0;31m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mImmediateResult\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfunc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    210\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    211\u001b[0m             \u001b[0mcallback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, batch)\u001b[0m\n\u001b[1;32m    588\u001b[0m         \u001b[0;31m# Don't delay the application, to avoid keeping the input\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    589\u001b[0m         \u001b[0;31m# arguments in memory\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 590\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    591\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    592\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    254\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    255\u001b[0m             return [func(*args, **kwargs)\n\u001b[0;32m--> 256\u001b[0;31m                     for func, args, kwargs in self.items]\n\u001b[0m\u001b[1;32m    257\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    258\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m    254\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    255\u001b[0m             return [func(*args, **kwargs)\n\u001b[0;32m--> 256\u001b[0;31m                     for func, args, kwargs in self.items]\n\u001b[0m\u001b[1;32m    257\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    258\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36m_logistic_regression_path\u001b[0;34m(X, y, pos_class, Cs, fit_intercept, max_iter, tol, verbose, solver, coef, class_weight, dual, penalty, intercept_scaling, multi_class, random_state, check_input, max_squared_sum, sample_weight, l1_ratio)\u001b[0m\n\u001b[1;32m    934\u001b[0m                 \u001b[0mfunc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mw0\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"L-BFGS-B\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mjac\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    935\u001b[0m                 \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1.\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0mC\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 936\u001b[0;31m                 \u001b[0moptions\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m\"iprint\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0miprint\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"gtol\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mtol\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"maxiter\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    937\u001b[0m             )\n\u001b[1;32m    938\u001b[0m             n_iter_i = _check_optimize_result(\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/_minimize.py\u001b[0m in \u001b[0;36mminimize\u001b[0;34m(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)\u001b[0m\n\u001b[1;32m    608\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mmeth\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'l-bfgs-b'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    609\u001b[0m         return _minimize_lbfgsb(fun, x0, args, jac, bounds,\n\u001b[0;32m--> 610\u001b[0;31m                                 callback=callback, **options)\n\u001b[0m\u001b[1;32m    611\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mmeth\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'tnc'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    612\u001b[0m         return _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/lbfgsb.py\u001b[0m in \u001b[0;36m_minimize_lbfgsb\u001b[0;34m(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, **unknown_options)\u001b[0m\n\u001b[1;32m    343\u001b[0m             \u001b[0;31m# until the completion of the current minimization iteration.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    344\u001b[0m             \u001b[0;31m# Overwrite f and g:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 345\u001b[0;31m             \u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfunc_and_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    346\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mtask_str\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstartswith\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mb'NEW_X'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    347\u001b[0m             \u001b[0;31m# new iteration\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/lbfgsb.py\u001b[0m in \u001b[0;36mfunc_and_grad\u001b[0;34m(x)\u001b[0m\n\u001b[1;32m    293\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    294\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0mfunc_and_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 295\u001b[0;31m             \u001b[0mf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    296\u001b[0m             \u001b[0mg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mjac\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    297\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mg\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/optimize.py\u001b[0m in \u001b[0;36mfunction_wrapper\u001b[0;34m(*wrapper_args)\u001b[0m\n\u001b[1;32m    325\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mfunction_wrapper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mwrapper_args\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    326\u001b[0m         \u001b[0mncalls\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 327\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mfunction\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mwrapper_args\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    328\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    329\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mncalls\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfunction_wrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/optimize.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, x, *args)\u001b[0m\n\u001b[1;32m     63\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__call__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     64\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnumpy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 65\u001b[0;31m         \u001b[0mfg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     66\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjac\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfg\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     67\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mfg\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36m_logistic_loss_and_grad\u001b[0;34m(w, X, y, alpha, sample_weight)\u001b[0m\n\u001b[1;32m    116\u001b[0m     \u001b[0mgrad\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mempty_like\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mw\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    117\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 118\u001b[0;31m     \u001b[0mw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0myz\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_intercept_dot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    119\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    120\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0msample_weight\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36m_intercept_dot\u001b[0;34m(w, X, y)\u001b[0m\n\u001b[1;32m     79\u001b[0m         \u001b[0mw\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mw\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     80\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 81\u001b[0;31m     \u001b[0mz\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msafe_sparse_dot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mw\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     82\u001b[0m     \u001b[0myz\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mz\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     83\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0myz\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/extmath.py\u001b[0m in \u001b[0;36msafe_sparse_dot\u001b[0;34m(a, b, dense_output)\u001b[0m\n\u001b[1;32m    149\u001b[0m             \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    150\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 151\u001b[0;31m         \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0ma\u001b[0m \u001b[0;34m@\u001b[0m \u001b[0mb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    152\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    153\u001b[0m     if (sparse.issparse(a) and sparse.issparse(b)\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# With our model tuned, lets test it out!\n",
    "LR_model = LogisticRegression(C=10000, random_state=1, max_iter=10000)\n",
    "LR_model.fit(X_train, y_train)\n",
    "\n",
    "print(f'The test accuracy is: {LR_model.score(X_test,y_test):0.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The test accuracy is: 0.912\n",
    "- Woah! That did something great for our accuracy! Lets do one more just for fun! What about the number of reviews a person gives? Do people who review just once do so if they are really upset? Let's say if a person reviews less than twice they are a small reviewer and if they are more than or equal to two, they are a small reviewer!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "def small_reviewer(Total_Number_of_Reviews_Reviewer_Has_Given):\n",
    "    if Total_Number_of_Reviews_Reviewer_Has_Given > 2:\n",
    "        return (0)\n",
    "    else:\n",
    "        return (1)\n",
    "    \n",
    "# df_test = df_train.apply(lambda x: func(x['col1'],x['col2']),axis=1)\n",
    "df_new = df_train.apply(lambda x: small_reviewer(x['Total_Number_of_Reviews_Reviewer_Has_Given']), axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [],
   "source": [
    "# pd.concat([df1, df3], sort=False)\n",
    "df_train_new = pd.concat([df_train, df_new], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train_new = df_train_new.rename(columns = {0: \"Small_Reviewer\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Additional_Number_of_Scoring</th>\n",
       "      <th>Average_Score</th>\n",
       "      <th>Review_Total_Negative_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews</th>\n",
       "      <th>Review_Total_Positive_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>lat</th>\n",
       "      <th>lng</th>\n",
       "      <th>Review_Month</th>\n",
       "      <th>...</th>\n",
       "      <th>p_world</th>\n",
       "      <th>p_worth</th>\n",
       "      <th>p_wouldn</th>\n",
       "      <th>p_year</th>\n",
       "      <th>p_years</th>\n",
       "      <th>p_yes</th>\n",
       "      <th>p_young</th>\n",
       "      <th>p_yummy</th>\n",
       "      <th>Reviewer_Score</th>\n",
       "      <th>Small_Reviewer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>220.0</td>\n",
       "      <td>9.1</td>\n",
       "      <td>20.0</td>\n",
       "      <td>902.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>275.0</td>\n",
       "      <td>51.494308</td>\n",
       "      <td>-0.175558</td>\n",
       "      <td>11.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1190.0</td>\n",
       "      <td>7.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5180.0</td>\n",
       "      <td>23.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>481.0</td>\n",
       "      <td>51.514879</td>\n",
       "      <td>-0.160650</td>\n",
       "      <td>4.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.425849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>299.0</td>\n",
       "      <td>8.3</td>\n",
       "      <td>81.0</td>\n",
       "      <td>1361.0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>672.0</td>\n",
       "      <td>51.521009</td>\n",
       "      <td>-0.123097</td>\n",
       "      <td>10.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>87.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>355.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>412.0</td>\n",
       "      <td>51.499749</td>\n",
       "      <td>-0.161524</td>\n",
       "      <td>6.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>317.0</td>\n",
       "      <td>7.6</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1458.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>499.0</td>\n",
       "      <td>51.516114</td>\n",
       "      <td>-0.174952</td>\n",
       "      <td>3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2588 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Additional_Number_of_Scoring  Average_Score  \\\n",
       "0                         220.0            9.1   \n",
       "1                        1190.0            7.5   \n",
       "2                         299.0            8.3   \n",
       "3                          87.0            9.0   \n",
       "4                         317.0            7.6   \n",
       "\n",
       "   Review_Total_Negative_Word_Counts  Total_Number_of_Reviews  \\\n",
       "0                               20.0                    902.0   \n",
       "1                                5.0                   5180.0   \n",
       "2                               81.0                   1361.0   \n",
       "3                               17.0                    355.0   \n",
       "4                               14.0                   1458.0   \n",
       "\n",
       "   Review_Total_Positive_Word_Counts  \\\n",
       "0                               21.0   \n",
       "1                               23.0   \n",
       "2                               27.0   \n",
       "3                               13.0   \n",
       "4                                0.0   \n",
       "\n",
       "   Total_Number_of_Reviews_Reviewer_Has_Given  days_since_review        lat  \\\n",
       "0                                         1.0              275.0  51.494308   \n",
       "1                                         6.0              481.0  51.514879   \n",
       "2                                         4.0              672.0  51.521009   \n",
       "3                                         7.0              412.0  51.499749   \n",
       "4                                         1.0              499.0  51.516114   \n",
       "\n",
       "        lng  Review_Month  ...  p_world  p_worth  p_wouldn  p_year   p_years  \\\n",
       "0 -0.175558          11.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "1 -0.160650           4.0  ...      0.0      0.0       0.0     0.0  0.425849   \n",
       "2 -0.123097          10.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "3 -0.161524           6.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "4 -0.174952           3.0  ...      0.0      0.0       0.0     0.0  0.000000   \n",
       "\n",
       "   p_yes  p_young  p_yummy  Reviewer_Score  Small_Reviewer  \n",
       "0    0.0      0.0      0.0             1.0               1  \n",
       "1    0.0      0.0      0.0             1.0               0  \n",
       "2    0.0      0.0      0.0             0.0               0  \n",
       "3    0.0      0.0      0.0             1.0               0  \n",
       "4    0.0      0.0      0.0             0.0               1  \n",
       "\n",
       "[5 rows x 2588 columns]"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train_new.head()\n",
    "# Looks like it's working!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting up our variables (With the new df)\n",
    "y_train = df_train_new['Reviewer_Score']\n",
    "X_train = df_train_new.drop(['Reviewer_Score'], axis=1)\n",
    "\n",
    "# Sub-sampling our data for speedy test and checks\n",
    "X_train = X_train.sample(frac=.1, random_state=42)\n",
    "y_train = y_train.sample(frac=.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11014    0.0\n",
       "11904    1.0\n",
       "4963     0.0\n",
       "4003     0.0\n",
       "8114     1.0\n",
       "Name: Reviewer_Score, dtype: float64"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Additional_Number_of_Scoring</th>\n",
       "      <th>Average_Score</th>\n",
       "      <th>Review_Total_Negative_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews</th>\n",
       "      <th>Review_Total_Positive_Word_Counts</th>\n",
       "      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>lat</th>\n",
       "      <th>lng</th>\n",
       "      <th>Review_Month</th>\n",
       "      <th>...</th>\n",
       "      <th>p_working</th>\n",
       "      <th>p_world</th>\n",
       "      <th>p_worth</th>\n",
       "      <th>p_wouldn</th>\n",
       "      <th>p_year</th>\n",
       "      <th>p_years</th>\n",
       "      <th>p_yes</th>\n",
       "      <th>p_young</th>\n",
       "      <th>p_yummy</th>\n",
       "      <th>Small_Reviewer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>11014</th>\n",
       "      <td>235.0</td>\n",
       "      <td>8.2</td>\n",
       "      <td>39.0</td>\n",
       "      <td>1003.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>436.0</td>\n",
       "      <td>51.491908</td>\n",
       "      <td>-0.168440</td>\n",
       "      <td>5.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11904</th>\n",
       "      <td>524.0</td>\n",
       "      <td>8.2</td>\n",
       "      <td>28.0</td>\n",
       "      <td>2516.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>77.0</td>\n",
       "      <td>51.498123</td>\n",
       "      <td>-0.179969</td>\n",
       "      <td>5.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4963</th>\n",
       "      <td>125.0</td>\n",
       "      <td>8.3</td>\n",
       "      <td>122.0</td>\n",
       "      <td>687.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>76.0</td>\n",
       "      <td>51.514033</td>\n",
       "      <td>-0.132065</td>\n",
       "      <td>5.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4003</th>\n",
       "      <td>791.0</td>\n",
       "      <td>7.3</td>\n",
       "      <td>41.0</td>\n",
       "      <td>3609.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>392.0</td>\n",
       "      <td>51.515367</td>\n",
       "      <td>-0.178327</td>\n",
       "      <td>7.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8114</th>\n",
       "      <td>620.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1974.0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>502.0</td>\n",
       "      <td>51.506558</td>\n",
       "      <td>-0.004514</td>\n",
       "      <td>3.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2587 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       Additional_Number_of_Scoring  Average_Score  \\\n",
       "11014                         235.0            8.2   \n",
       "11904                         524.0            8.2   \n",
       "4963                          125.0            8.3   \n",
       "4003                          791.0            7.3   \n",
       "8114                          620.0            9.0   \n",
       "\n",
       "       Review_Total_Negative_Word_Counts  Total_Number_of_Reviews  \\\n",
       "11014                               39.0                   1003.0   \n",
       "11904                               28.0                   2516.0   \n",
       "4963                               122.0                    687.0   \n",
       "4003                                41.0                   3609.0   \n",
       "8114                                 0.0                   1974.0   \n",
       "\n",
       "       Review_Total_Positive_Word_Counts  \\\n",
       "11014                               41.0   \n",
       "11904                               22.0   \n",
       "4963                                 0.0   \n",
       "4003                                13.0   \n",
       "8114                                24.0   \n",
       "\n",
       "       Total_Number_of_Reviews_Reviewer_Has_Given  days_since_review  \\\n",
       "11014                                         2.0              436.0   \n",
       "11904                                         4.0               77.0   \n",
       "4963                                          1.0               76.0   \n",
       "4003                                          2.0              392.0   \n",
       "8114                                          1.0              502.0   \n",
       "\n",
       "             lat       lng  Review_Month  ...  p_working  p_world  p_worth  \\\n",
       "11014  51.491908 -0.168440           5.0  ...        0.0      0.0      0.0   \n",
       "11904  51.498123 -0.179969           5.0  ...        0.0      0.0      0.0   \n",
       "4963   51.514033 -0.132065           5.0  ...        0.0      0.0      0.0   \n",
       "4003   51.515367 -0.178327           7.0  ...        0.0      0.0      0.0   \n",
       "8114   51.506558 -0.004514           3.0  ...        0.0      0.0      0.0   \n",
       "\n",
       "       p_wouldn  p_year  p_years  p_yes  p_young  p_yummy  Small_Reviewer  \n",
       "11014       0.0     0.0      0.0    0.0      0.0      0.0               1  \n",
       "11904       0.0     0.0      0.0    0.0      0.0      0.0               0  \n",
       "4963        0.0     0.0      0.0    0.0      0.0      0.0               1  \n",
       "4003        0.0     0.0      0.0    0.0      0.0      0.0               1  \n",
       "8114        0.0     0.0      0.0    0.0      0.0      0.0               1  \n",
       "\n",
       "[5 rows x 2587 columns]"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting out 20% for test set\n",
    "X_remainder, X_test, y_remainder, y_test = train_test_split(X_train, y_train, test_size = 0.2,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Finished cross validation with c = 100000000000.0..\r"
     ]
    }
   ],
   "source": [
    "#Store the results\n",
    "cross_validation_scores = []\n",
    "\n",
    "C_range = np.array([.00000001,.0000001,.000001,.00001,.0001,.001,.1,\\\n",
    "                1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000,10000000000,100000000000])\n",
    "\n",
    "#Do some cross validation\n",
    "for c in C_range:\n",
    "    LR_model = LogisticRegression(C=c,random_state=1)\n",
    "    \n",
    "    # the cross validation score (mean of scores from all folds)\n",
    "    cv_score = np.mean(cross_val_score(LR_model, X_remainder, y_remainder, cv = 5))\n",
    "    \n",
    "    cross_validation_scores.append(cv_score)\n",
    "    print(f'Finished cross validation with c = {c}..', end='\\r')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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K3wZPHgwmZmZLFiwgMcff5wrrriCJ554glWrVrF48WJmz57Nn/70Jz7//PNvrFtVzZSYmFhdvQTw0EMP8cgjj3D99deTnp7OjBkzao1RVWu9xfOPf/wjiYmJrF+/Ho/HU/3B7I+v46zafps2p78YRkZGUlFR4bNsWloaaWlpXH755dx999088sg3OrSudV9Vap77u+66i9/+9rd1xl8f1gZB4+tsP954gJ//fT2DE+N45PLBda9gTJCM6duZWfeN55ErhjTJCIWXXHIJZWVlvPLKK9XzMjIyfH6YT5w4kXnz5nHixAlKSkp4//33mTBhAvv376ddu3Z897vf5dFHH2XNmjUUFxdTWFjI1VdfzXPPPVddj1/TzTffzIIFC5gzZw633XZb9fzCwkJ693aGqX/jjTf8HkOnTp3o2LEjy5cvB2DWrFlnbKdnz55ERETw5ptvVtfbd+jQgaKiIp/bmzhxYvU20tPT6datG/Hx8X5jqLJ//37WrFlTPb1u3Tr69u1LfHw8SUlJzJs3D4CysjJOnDjBxIkTmTNnDpWVlRw5coSlS5f6bKS/9NJLmTt3LocPHwacdpK9e/cGFJM/rf4KYln2Eaa9lkGlKpEiXJTcla7tA68eyi8pY+n2PBTYm3+CDTmFNgiQCakxfTs32d+giPD+++/z05/+lKeffprY2Fj69evHc889R25u7hllR48ezbRp06o/wO677z5GjRrFwoULmT59OhEREURHR/PSSy9RVFTEDTfcQGlpKapa6zffTp06MX78eA4dOnRGNdKMGTP4zne+Q+/evRk/fjy7d+/2exyvv/4699xzD+3atePKK6+snv+jH/2Im2++mXfffZfJkydXfysfPnw4UVFRjBgxgmnTpjFq1Kgz9n333XczfPhw2rVrV2eC8lZeXs6jjz7K/v37iY2NpXv37vz5z38G4M033+QHP/gBTzzxBNHR0bz77rtMmTKFFStWMGLECESEZ555hh49erB169Yztjt06FB+/etfc8UVV+DxeIiOjmbmzJn07ds34Nh8EX+XMC1Jamqqrl69ut7rPbtwGy8u2VE93bFtFPFtA6+zPX6ynMKTzmVkpMAjVwzhx5MH1TuOlj7oucUfHFu2bOHcc8+ts1y49wVUF4u/efj6exKRTFX1eQ9vq7+CmPytBF5dvqu6zva1aWn1+vZVs853/ICu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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(C_range, cross_validation_scores,label=\"Cross Validation Score\",marker='.')\n",
    "plt.legend()\n",
    "plt.xscale(\"log\")\n",
    "plt.xlabel('Regularization Parameter: C')\n",
    "plt.ylabel('Cross Validation Score')\n",
    "plt.grid()\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- C = 10 seems to yield the highest cross validation score!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-95-5628252447be>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#With our model tuned, lets test it out!\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mLR_model\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mLogisticRegression\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mC\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10000\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mLR_model\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf'The test accuracy is: {LR_model.score(X_test, y_test):0.3f}'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m   1599\u001b[0m                       \u001b[0mpenalty\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mpenalty\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmax_squared_sum\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmax_squared_sum\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1600\u001b[0m                       sample_weight=sample_weight)\n\u001b[0;32m-> 1601\u001b[0;31m             for class_, warm_start_coef_ in zip(classes_, warm_start_coef))\n\u001b[0m\u001b[1;32m   1602\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1603\u001b[0m         \u001b[0mfold_coefs_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_iter_\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mfold_coefs_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, iterable)\u001b[0m\n\u001b[1;32m   1002\u001b[0m             \u001b[0;31m# remaining jobs.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1003\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1004\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdispatch_one_batch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1005\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_original_iterator\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1006\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36mdispatch_one_batch\u001b[0;34m(self, iterator)\u001b[0m\n\u001b[1;32m    833\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    834\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 835\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_dispatch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtasks\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    836\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    837\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m_dispatch\u001b[0;34m(self, batch)\u001b[0m\n\u001b[1;32m    752\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    753\u001b[0m             \u001b[0mjob_idx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_jobs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 754\u001b[0;31m             \u001b[0mjob\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mapply_async\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    755\u001b[0m             \u001b[0;31m# A job can complete so quickly than its callback is\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    756\u001b[0m             \u001b[0;31m# called before we get here, causing self._jobs to\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py\u001b[0m in \u001b[0;36mapply_async\u001b[0;34m(self, func, callback)\u001b[0m\n\u001b[1;32m    207\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mapply_async\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    208\u001b[0m         \u001b[0;34m\"\"\"Schedule a func to be run\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 209\u001b[0;31m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mImmediateResult\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfunc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    210\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    211\u001b[0m             \u001b[0mcallback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, batch)\u001b[0m\n\u001b[1;32m    588\u001b[0m         \u001b[0;31m# Don't delay the application, to avoid keeping the input\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    589\u001b[0m         \u001b[0;31m# arguments in memory\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 590\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    591\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    592\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    254\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    255\u001b[0m             return [func(*args, **kwargs)\n\u001b[0;32m--> 256\u001b[0;31m                     for func, args, kwargs in self.items]\n\u001b[0m\u001b[1;32m    257\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    258\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/joblib/parallel.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m    254\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    255\u001b[0m             return [func(*args, **kwargs)\n\u001b[0;32m--> 256\u001b[0;31m                     for func, args, kwargs in self.items]\n\u001b[0m\u001b[1;32m    257\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    258\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36m_logistic_regression_path\u001b[0;34m(X, y, pos_class, Cs, fit_intercept, max_iter, tol, verbose, solver, coef, class_weight, dual, penalty, intercept_scaling, multi_class, random_state, check_input, max_squared_sum, sample_weight, l1_ratio)\u001b[0m\n\u001b[1;32m    934\u001b[0m                 \u001b[0mfunc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mw0\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"L-BFGS-B\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mjac\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    935\u001b[0m                 \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1.\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0mC\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 936\u001b[0;31m                 \u001b[0moptions\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m\"iprint\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0miprint\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"gtol\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mtol\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"maxiter\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    937\u001b[0m             )\n\u001b[1;32m    938\u001b[0m             n_iter_i = _check_optimize_result(\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/_minimize.py\u001b[0m in \u001b[0;36mminimize\u001b[0;34m(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)\u001b[0m\n\u001b[1;32m    608\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mmeth\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'l-bfgs-b'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    609\u001b[0m         return _minimize_lbfgsb(fun, x0, args, jac, bounds,\n\u001b[0;32m--> 610\u001b[0;31m                                 callback=callback, **options)\n\u001b[0m\u001b[1;32m    611\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mmeth\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'tnc'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    612\u001b[0m         return _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/lbfgsb.py\u001b[0m in \u001b[0;36m_minimize_lbfgsb\u001b[0;34m(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, **unknown_options)\u001b[0m\n\u001b[1;32m    343\u001b[0m             \u001b[0;31m# until the completion of the current minimization iteration.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    344\u001b[0m             \u001b[0;31m# Overwrite f and g:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 345\u001b[0;31m             \u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfunc_and_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    346\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mtask_str\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstartswith\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mb'NEW_X'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    347\u001b[0m             \u001b[0;31m# new iteration\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/lbfgsb.py\u001b[0m in \u001b[0;36mfunc_and_grad\u001b[0;34m(x)\u001b[0m\n\u001b[1;32m    293\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    294\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0mfunc_and_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 295\u001b[0;31m             \u001b[0mf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    296\u001b[0m             \u001b[0mg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mjac\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    297\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mg\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/optimize.py\u001b[0m in \u001b[0;36mfunction_wrapper\u001b[0;34m(*wrapper_args)\u001b[0m\n\u001b[1;32m    325\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mfunction_wrapper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mwrapper_args\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    326\u001b[0m         \u001b[0mncalls\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 327\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mfunction\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mwrapper_args\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    328\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    329\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mncalls\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfunction_wrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/scipy/optimize/optimize.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, x, *args)\u001b[0m\n\u001b[1;32m     63\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__call__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     64\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnumpy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 65\u001b[0;31m         \u001b[0mfg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     66\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjac\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfg\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     67\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mfg\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36m_logistic_loss_and_grad\u001b[0;34m(w, X, y, alpha, sample_weight)\u001b[0m\n\u001b[1;32m    116\u001b[0m     \u001b[0mgrad\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mempty_like\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mw\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    117\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 118\u001b[0;31m     \u001b[0mw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0myz\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_intercept_dot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    119\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    120\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0msample_weight\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py\u001b[0m in \u001b[0;36m_intercept_dot\u001b[0;34m(w, X, y)\u001b[0m\n\u001b[1;32m     79\u001b[0m         \u001b[0mw\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mw\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     80\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 81\u001b[0;31m     \u001b[0mz\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msafe_sparse_dot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mw\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     82\u001b[0m     \u001b[0myz\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mz\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     83\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0myz\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/extmath.py\u001b[0m in \u001b[0;36msafe_sparse_dot\u001b[0;34m(a, b, dense_output)\u001b[0m\n\u001b[1;32m    149\u001b[0m             \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    150\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 151\u001b[0;31m         \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0ma\u001b[0m \u001b[0;34m@\u001b[0m \u001b[0mb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    152\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    153\u001b[0m     if (sparse.issparse(a) and sparse.issparse(b)\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "#With our model tuned, lets test it out!\n",
    "LR_model = LogisticRegression(C=10, random_state=1, max_iter=10000)\n",
    "LR_model.fit(X_train, y_train)\n",
    "\n",
    "print(f'The test accuracy is: {LR_model.score(X_test, y_test):0.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The test accuracy is: 0.938\n",
    "- Looks like the number of reviews a revewier leaves is another good tool for accuracy!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "toc-autonumbering": false
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
