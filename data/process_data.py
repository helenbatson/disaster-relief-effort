import re
import sys
import numpy as np
import pandas as pd
import sqlite3
import sqlalchemy as db
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load message and category files and merge into a dataframe
    Latin-1 represents the alphabets of Western European languages.
    '''
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    '''
    Split `categories into separate category columns.
    - Split the values in the `categories` column on the `;`
    character so that each value becomes a separate column.
    - Use the first row of categories dataframe to create column names for the categories data.
    - Rename columns of `categories` with new column names.
    - Input zeros and ones into the dataframe.
    - Return df with new category columns.
    '''
    categories = df.categories.str.split(';',expand=True)

    # Use to split into separate columns
    subs = df.categories.iloc[0].split(';')
    subs = pd.Series(subs)

    def cols(word):
        x = re.findall("[^\d\W]+", word)
        if x :
            return(x)

    new_columns = subs.apply(cols)

    # Use the row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing

    category_colnames = []

    for newcol in new_columns.str[0]:
        category_colnames.append(newcol)

    categories.columns = category_colnames


    # For each column in the df, ensure 0 / 1 in column values; replace 2s with 1s
    def create_booleans(df):
        for column in df.columns:
            zero_one = []

            # set each value to be the last character of the string
            # convert column from string to numeric
            for value in df[column]:
                zero_one.append(int(value[-1:]))

            # Change value 2 to value 1 for boolean values throughout the matrix
            zero_one = np.array(zero_one)
            zero_one = np.where(zero_one==2, 1, zero_one)

            df[column]=zero_one

    create_booleans(categories)

    # Replace `categories` column in `df` with new category columns.
    #- Drop the categories column from the df dataframe since it is no longer needed.
    #- Concatenate df and categories data frames.

    df = pd.concat([df, categories], axis=1).reindex(df.index)


    #Remove duplicates
    df=df.drop_duplicates()

    # drop the id column as it will not be needed in ML pipelines
    df=df.drop(['id'], axis=1)

    # return clean df
    return df


def save_data(df, database_filename):
    '''
    Save df to database as a table: disaster_table
    '''
    # Save the clean dataset into an sqlite database.
    # Create the connection
    # Set echo=True to see all of the output that comes from our database connection.

    root = 'sqlite:///'
    engine = create_engine(root+database_filename, echo=True)

    sqlite_connection = engine.connect()

    sqlite_table = "disaster_table"
    df.to_sql(sqlite_table, sqlite_connection, if_exists='fail')

    sqlite_connection.close()

def main():
    '''
    Run all functions to load the data, clean the data and save it to a database.
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
