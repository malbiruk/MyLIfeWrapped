'''
save events' data from google calendar as dataframe
'''

import argparse
import datetime
import os.path
from dataclasses import dataclass

import googlemaps
import pandas as pd
from calendar_.my_api_keys import GM_API_KEY
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from pandas import DataFrame
from progress_bar import progress_bar

# from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
CALENDARS_TO_EXCLUDE = ['Birthdays',
                        'k.kostyuk@insilicomedicine.com',
                        'k.kostyuk@easyomics.com',
                        ]

DIR_PATH = 'calendar_'


def create_credentials(scopes: list, directory: str = '.') -> None:
    '''
    generate token.json for user and use it if it exists
    '''
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(f'{directory}/token.json'):
        creds = Credentials.from_authorized_user_file(
            f'{directory}/token.json', scopes)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                f'{directory}/credentials.json', scopes
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(f'{directory}/token.json', 'w', encoding='utf-8') as token:
            token.write(creds.to_json())
    return creds


@dataclass
class Calendar:
    '''
    class for keeping calendars' info
    '''
    name: str
    id: str
    color: str
    primary: bool
    timezone: str
    events: DataFrame = None

    def __repr__(self) -> None:
        '''
        returns a string containing only name attribute
        '''
        return f'Calendar({self.name})'

    def get_events(self, service: Resource, start_time: str) -> DataFrame:
        '''
        create dataframe of events
        '''
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        page_token = None
        all_events = []
        while True:
            events = service.events().list(
                calendarId=self.id,
                timeMin=start_time,
                timeMax=now,
                singleEvents=True,
                orderBy='startTime').execute()
            for event in events.get('items'):
                # without full day events
                if event.get('start').get('dateTime'):
                    all_events.append(
                        {
                            'category': self.name,
                            'category_color': self.color,
                            'event_name': event.get('summary'),
                            'description': event.get('description'),
                            'location': event.get('location'),
                            'start': event.get('start').get('dateTime'),
                            'end': event.get('end').get('dateTime')}
                    )

            page_token = events.get('nextPageToken')
            if not page_token:
                break
        all_events = pd.DataFrame(all_events)
        self.events = all_events
        return all_events


def get_calendars(service: Resource) -> list:
    '''
    returns list of calendars (Calendar class)
    '''
    my_calendars = []
    page_token = None
    while True:
        calendar_list = service.calendarList().list(
            pageToken=page_token).execute()
        for calendar in calendar_list['items']:
            my_calendars.append(
                Calendar(
                    calendar['summary'],
                    calendar['id'],
                    calendar['backgroundColor'],
                    calendar['primary'] if calendar.get('primary') else False,
                    calendar['timeZone']))
        page_token = calendar_list.get('nextPageToken')
        if not page_token:
            break
    return my_calendars


def get_daily_notes(calendar: Calendar,
                    service: Resource,
                    start_time: str) -> DataFrame:
    '''
    save mood and daily notes to its own csv file
    '''
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    page_token = None
    all_events = []
    while True:
        events = service.events().list(
            calendarId=calendar.id,
            timeMin=start_time,
            timeMax=now,
            singleEvents=True,
            orderBy='startTime').execute()
        for event in events.get('items'):
            # without full day events
            if event.get('start').get('date'):
                mood = (event.get('description')
                        .split('\n', 1)[0]
                        .removeprefix('mood: '))
                notes = (event.get('description')
                         .split('\n', 1)[1]
                         .removeprefix('notes: '))
                all_events.append(
                    {
                        'date': event.get('start').get('date'),
                        'mood': mood,
                        'notes': notes
                    }
                )
        page_token = events.get('nextPageToken')
        if not page_token:
            break
    return pd.DataFrame(all_events)


COLORS_DICT = {
    'Cocoa': {'Classic': '#AC725E', 'Modern': '#795548'},
    'Flamingo': {'Classic': '#D06B64', 'Modern': '#E67C73'},
    'Tomato': {'Classic': '#F83A22', 'Modern': '#D50000'},
    'Tangerine': {'Classic': '#FA573C', 'Modern': '#F4511E'},
    'Pumpkin': {'Classic': '#FF7537', 'Modern': '#EF6C00'},
    'Mango': {'Classic': '#FFAD46', 'Modern': '#F09300'},
    'Eucalyptus': {'Classic': '#42D692', 'Modern': '#009688'},
    'Basil': {'Classic': '#16A765', 'Modern': '#0B8043'},
    'Pistachio': {'Classic': '#7BD148', 'Modern': '#7CB342'},
    'Avocado': {'Classic': '#B3DC6C', 'Modern': '#C0CA33'},
    'Citron': {'Classic': '#FBE983', 'Modern': '#E4C441'},
    'Banana': {'Classic': '#FAD165', 'Modern': '#F6BF26'},
    'Sage': {'Classic': '#92E1C0', 'Modern': '#33B679'},
    'Peacock': {'Classic': '#9FE1E7', 'Modern': '#039BE5'},
    'Cobalt': {'Classic': '#9FC6E7', 'Modern': '#4285F4'},
    'Blueberry': {'Classic': '#4986E7', 'Modern': '#3F51B5'},
    'Lavender': {'Classic': '#9A9CFF', 'Modern': '#7986CB'},
    'Wisteria': {'Classic': '#B99AFF', 'Modern': '#B39DDB'},
    'Graphite': {'Classic': '#C2C2C2', 'Modern': '#616161'},
    'Birch': {'Classic': '#CABDBF', 'Modern': '#A79B8E'},
    'Radicchio': {'Classic': '#CCA6AC', 'Modern': '#AD1457'},
    'Cherry Blossom': {'Classic': '#F691B2', 'Modern': '#D81B60'},
    'Grape': {'Classic': '#CD74E6', 'Modern': '#8E24AA'},
    'Amethyst': {'Classic': '#A47AE2', 'Modern': '#9E69AF'}
}

COLOR_CLASSIC_TO_MODERN = {v['Classic']: v['Modern']
                           for k, v in COLORS_DICT.items()}

OUTSIDE_EVENTS = ['walk', 'bus', 'metro', 'taxi', 'taking out the trash',
                  'rollerblading']


def service_geocode(gmaps: googlemaps.Client, address: str) -> tuple:
    '''
    simple function to apply geocode to get coordinates from address
    '''
    location = gmaps.geocode(address)
    if location is not None:
        return (location[0]['geometry']['location']['lat'],
                location[0]['geometry']['location']['lng'])
    return None


def geocode_unique_addresses(df: DataFrame,
                             gmaps: googlemaps.Client) -> DataFrame:
    '''
    geocode only uniaue addresses from df and then merge obtained coords
    back to initial df
    '''
    # Step 1: Create a new DataFrame with unique addresses
    unique_addresses = df['location'].unique()
    unique_df = pd.DataFrame({'location': unique_addresses})

    # Step 2: Apply geocoding function to obtain coordinates
    unique_df['coordinates'] = unique_df['location'].apply(
        lambda x: service_geocode(gmaps, x) if x is not None else None)

    # Step 3: Merge coordinates back to the original DataFrame
    df = pd.merge(df, unique_df, on='location', how='left')

    return df


def preprocess_df(df: DataFrame, timezone) -> DataFrame:
    '''
    1. convert datetime columns to datetime
    2. add duration column and local start an end datetime columns
    3. strip() all str values
    4. convert colors to modern
    5. add 'outside' to description of outside events
    6. geocode locations
    '''
    # 1. convert datetime columns to datetime
    df.start = pd.to_datetime(df.start, utc=True)
    df.end = pd.to_datetime(df.end, utc=True)

    # 2. add duration column and local start an end datetime columns
    df['duration'] = df.end - df.start
    df['start_local'] = pd.to_datetime(df.start).dt.tz_convert(timezone)
    df['end_local'] = pd.to_datetime(df.end).dt.tz_convert(timezone)

    # 3. strip() all str values
    df_obj = df.select_dtypes('object')
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # 4. convert classic colors to modern
    df.category_color = df.category_color.str.upper().apply(
        lambda x: COLOR_CLASSIC_TO_MODERN.get(x, x))

    # 5. add 'outside' to description of outside events
    df.description = df.description.fillna('')
    to_outside_bool = (
        ((df['event_name'].isin(OUTSIDE_EVENTS))
            | (~df['location'].isna()))
        & (~df['description'].str.startswith('outside')))
    df.loc[to_outside_bool, 'description'] = (
        'outside\n' + df.loc[to_outside_bool, 'description']).str.strip()

    # 6. geocode locations
    gmaps = googlemaps.Client(GM_API_KEY)
    return geocode_unique_addresses(df, gmaps)


def main(force: bool) -> None:
    '''
    save events data from google calendar as dataframe
    (events from 2024-02-01 or last saved to now)
    '''
    start_time = '2024-02-01T00:00:00Z'
    print('getting credentials...')
    creds = create_credentials(
        ['https://www.googleapis.com/auth/calendar.readonly'], DIR_PATH)
    service = build('calendar', 'v3', credentials=creds)

    print('downloading data...')
    my_calendars = get_calendars(service)
    my_calendars = [i for i in my_calendars
                    if i.name not in CALENDARS_TO_EXCLUDE]
    primary_calendar = [i for i in my_calendars if i.primary][0]
    timezone = primary_calendar.timezone
    my_calendars = [i for i in my_calendars if i != primary_calendar]

    if os.path.exists(f'{DIR_PATH}/all_events.csv') and not force:
        prev_df = pd.read_csv(f'{DIR_PATH}/all_events.csv')
        start_time = prev_df.iloc[-1]['start']
        for from_, to_ in {' ': 'T', '+00:00': 'Z'}.items():
            start_time = start_time.replace(from_, to_)

    calendar_dfs = []
    with progress_bar as p:
        for cal in p.track(my_calendars,
                           description='dnld calendars'):
            calendar_dfs.append(cal.get_events(service, start_time))

    df = pd.concat(calendar_dfs)
    df = preprocess_df(df, timezone)

    if os.path.exists(f'{DIR_PATH}/all_events.csv') and not force:
        cols = ['start', 'end', 'start_local', 'end_local']
        prev_df[cols] = prev_df[cols].apply(pd.to_datetime)
        prev_df.duration = pd.to_timedelta(prev_df.duration)
        df = pd.concat([prev_df, df])

    df.sort_values(by='start', inplace=True)
    df.drop_duplicates(inplace=True)

    df.to_csv(f'{DIR_PATH}/all_events.csv', index=False)

    other_cal = [i for i in my_calendars if i.name == 'other'][0]
    dn = get_daily_notes(other_cal, service, '2024-02-01T00:00:00Z')
    dn.to_csv(f'{DIR_PATH}/daily_notes.csv', index=False)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="save events' data from google calendar as .csv"
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='replace old .csv file and download all events (not only new)')
    args = parser.parse_args()
    main(**vars(args))
