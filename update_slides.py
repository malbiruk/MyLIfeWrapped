import pandas as pd
from calendar_.obtain_data import create_credentials
from googleapiclient.discovery import build


def create_slide(
        service,
        presentation_id,
        image_url,
        page_id,
        page_idx):
    requests = [
        {
            'deleteObject': {
                "objectId": page_id,
            }
        },
        {
            "createSlide": {
                "insertionIndex": str(page_idx),
                "objectId": page_id,
            }
        },
        {
            "updatePageProperties": {
                "objectId": page_id,
                "pageProperties": {
                    "pageBackgroundFill": {
                        "stretchedPictureFill": {
                            "contentUrl": image_url,
                        }
                    }
                },
                "fields": "pageBackgroundFill"
            }
        },
        {
            "createShape": {
                "objectId": f'{page_id}_bck',
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "height": {"magnitude": 16*72, "unit": "PT"},
                        "width": {"magnitude": 4.5*72, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 0,
                        "translateY": 0,
                        "unit": "PT"
                    }
                }
            }
        },
        {
            "createShape": {
                "objectId": f'{page_id}_fwd',
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": page_id,
                    "size": {
                        "height": {"magnitude": 16*72, "unit": "PT"},
                        "width": {"magnitude": 4.5*72, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 4.5*72,
                        "translateY": 0,
                        "unit": "PT"
                    }
                }
            }
        },
        {
            "updateShapeProperties": {
                "objectId": f'{page_id}_fwd',
                "fields": "link",
                "shapeProperties": {
                    "link": {
                        "relativeLink": 'NEXT_SLIDE',
                    }
                }
            }
        },
        {
            "updateShapeProperties": {
                "objectId": f'{page_id}_bck',
                "fields": "link",
                "shapeProperties": {
                    "link": {
                        "relativeLink": 'PREVIOUS_SLIDE',
                    }
                }
            }
        }
    ]

    # delete only if slide exists:
    slides = (
        service.presentations()
        .get(presentationId=presentation_id)
        .execute()
        .get("slides", [])
    )
    slide_exists = any(slide.get("objectId") == page_id for slide in slides)
    if not slide_exists:
        requests = requests[1:]
        print(f'{page_id} did not exist - adding new slide')

    # Execute the request.
    body = {"requests": requests}
    response = (
        service.presentations()
        .batchUpdate(presentationId=presentation_id, body=body)
        .execute()
    )
    create_slide_response = response.get("replies")[0].get("createSlide")


def main():
    creds = create_credentials(
        ['https://www.googleapis.com/auth/drive',
         # 'https://www.googleapis.com/auth/spreadsheets',
         'https://www.googleapis.com/auth/presentations'
         ])

    service = build('slides', 'v1', credentials=creds)

    df = pd.read_csv('img_ids.csv')
    df = df.sort_values('filename', ascending=False)
    df['obj_id'] = df.filename.str.replace('.png', '')
    df.link = 'https://drive.google.com/uc?id=' + df.id

    slide_idx = 3
    print('updating slides...')
    for tr in ['week', 'month', 'half-year', 'year'][::-1]:
        for fw in ['sfw', 'nsfw']:
            df_part = df.loc[df.filename.str.contains(f'_{tr}_{fw}')]
            for _, row in df_part.iterrows():
                create_slide(
                    service,
                    '1H430Yl3KYi__btPtraAcyNV0vw4Pp7__dkAfH3NE3GU',
                    row.link,
                    row.obj_id,
                    slide_idx)
            slide_idx += (len(df_part) + 2)

    print('done.')


if __name__ == '__main__':
    main()
