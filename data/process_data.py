import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files.
    Args:
        messages_filepath: path to csv file of messages
        categories_filepath: path to csv file of categories

    Returns:
        A merged dataframe of messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=['id'], how='left')
    return df


def clean_data(df):
    """
    Clean the dataframe and create new columns for categories.
    Args:
        df: The dataframe.
    Returns:
        Cleaned dataframe.
    """
    categories = df['categories'].str.split(';', expand=True)

    # Select one row to see categories
    row = categories[:1]
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0])
    # Rename columns of categories DataFrame
    categories.columns = category_colnames

    # Change value in df to be 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """

    Args:
        df: dataframe
        database_filename: filename where database needs to be dave

    Returns:

    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False)


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