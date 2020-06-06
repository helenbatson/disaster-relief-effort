import re
import sys
import pandas as pd
import sqlite3
import sqlalchemy as db
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')

    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):

    '''
    Split `categories` into separate category columns.
    - Split the values in the `categories` column on the `;`
    character so that each value becomes a separate column.
    - Use the first row of categories dataframe to create column names for the categories data.
    - Rename columns of `categories` with new column names.
    '''
    categories = df.categories.str.split(';',expand=True)

    # Use to split into separate columns
    subs = df.categories.iloc[0].split(';')
    subs = pd.Series(subs)

    def cols(word):
        x = re.findall("[^\d\W]+", word)
        if x :
            return(x) # subs.apply(lambda x: cols(x))

    new_columns = subs.apply(cols)

    # Use the row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing

    category_colnames = []

    for newcol in new_columns.str[0]:
        category_colnames.append(newcol)

    categories.columns = category_colnames


    # for each column in the df, ensure 0 / 1 in column values; replace 2s with 1s
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

    #Replace `categories` column in `df` with new category columns.
    #- Drop the categories column from the df dataframe since it is no longer needed.
    #- Concatenate df and categories data frames.

    df = pd.merge(df.reset_index(drop=True),
                 categories.reset_index(drop=True),
                 left_index=True,
                 right_index=True)


    #Remove duplicates
    df=df.drop_duplicates()

    # drop the id column as it will not be needed in ML pipelines
    df=df.drop(['id'], axis=1)

    # return clean df
    return df


def save_data(df, database_filename):
    #Save the clean dataset into an sqlite database.
    # Create the connection
    # Set echo=True to see all of the output that comes from our database connection.
    engine = create_engine('sqlite:///DisasterResponse.db', echo=True)
    sqlite_connection = engine.connect()

    sqlite_table = "disaster_table"
    df.to_sql(sqlite_table, sqlite_connection, if_exists='fail')

    sqlite_connection.close()

def main():
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
