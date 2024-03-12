import os

import pandas as pd
from calendar_.obtain_data import create_credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from progress_bar import progress_bar

# pylint: disable=maybe-no-member

def create_folder(service, folder_name: str, parent_folder_id=None):
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id:
        folder_metadata['parents'] = [parent_folder_id]

    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')


def get_folder_id_by_name(service, folder_name, parent_folder_id=None):
    # Set up the query
    query = f"name = '{folder_name}' "\
        "and mimeType = 'application/vnd.google-apps.folder' "\
        "and trashed = false"
    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"

    # Search for the folder
    response = service.files().list(q=query).execute()

    # Grab that ID if it exists
    if 'files' in response and response['files']:
        folder_id = response['files'][0]['id']
        return folder_id
    else:
        return None


def get_file_id_by_name(service, file_name, parent_folder_id=None):
    # Set up the query
    query = f"name = '{file_name}' and trashed = false"
    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"

    # Search for the folder
    response = service.files().list(q=query).execute()

    # Grab that ID if it exists
    if 'files' in response and response['files']:
        file_id = response['files'][0]['id']
        return file_id
    return None


def upload_image(service, file_path, folder_id):
    media = MediaFileUpload(file_path)
    file_metadata = {
        'name': file_path.split("/")[-1],  # Assuming Unix-style path
        'parents': [folder_id]
    }
    return (service.files()
            .create(body=file_metadata, media_body=media, fields='id')
            .execute().get('id'))


def create_google_sheet(service, sheet_name, parent_folder_id):
    sheet_metadata = {
        'name': sheet_name,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
        'parents': [parent_folder_id]
    }

    sheet = service.files().create(body=sheet_metadata, fields='id').execute()
    return sheet.get('id')


def get_shareable_link(service, file_id):
    service.permissions().create(
        fileId=file_id,
        body={'role': 'reader', 'type': 'anyone'}
    ).execute()

    return f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'

def main():
    creds = create_credentials(
        ['https://www.googleapis.com/auth/drive',
         'https://www.googleapis.com/auth/presentations'])
    service = build('drive', 'v3', credentials=creds)

    # Check if 'MyLifeWrapped' exists, create if not
    my_life_wrapped_id = get_folder_id_by_name(service, 'MyLifeWrapped')
    if my_life_wrapped_id is None:
        my_life_wrapped_id = create_folder(service, 'MyLifeWrapped')
    else:
        pictures_folder_id = get_folder_id_by_name(
            service, 'pictures',
            parent_folder_id=my_life_wrapped_id)
        service.files().delete(fileId=pictures_folder_id).execute()

    pictures_folder_id = create_folder(service, 'pictures',
                                       parent_folder_id=my_life_wrapped_id)

    # Now, loop through your local folder and upload each image
    # print('uploading imgs...')
    df = []
    pictures = os.listdir('pictures')
    with progress_bar as p:
        for filename in p.track(pictures,
                                total=len(pictures),
                                description='uploading images'):
            file_path = os.path.join('pictures', filename)
            file_id = upload_image(service, file_path, pictures_folder_id)
            shareable_link = get_shareable_link(service, file_id)
            df.append(
            {'filename': filename,
            'id': file_id,
            'image': f'pictures/{filename}',
            'link': shareable_link})


    df = pd.DataFrame(df)


    df.to_csv('img_ids.csv', index=False)
    # print('done.')

if __name__ == '__main__':
    main()
