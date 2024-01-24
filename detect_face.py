import boto3
import io
from PIL import Image
from datetime import datetime
import pandas as pd

Attendance = pd.DataFrame(columns=['FullName', 'CurrentTime'])

# Function to update the DataFrame
def update_attendance(full_name, current_time):
    # Create a new DataFrame with the full name and current time
    new_data = pd.DataFrame({'FullName': [full_name], 'CurrentTime': [current_time]})
    # Append the new data to the Attendance DataFrame
    return Attendance._append(new_data, ignore_index=False)

rekognition = boto3.client('rekognition', region_name='us-east-1')
dynamodb = boto3.client('dynamodb', region_name='us-east-1')

image_path = input("Enter path of the image to check: ")

image = Image.open(image_path)
stream = io.BytesIO()
image.save(stream,format="JPEG")
image_binary = stream.getvalue()


response = rekognition.search_faces_by_image(
        CollectionId='Ramzy',
        Image={'Bytes':image_binary}                                       
        )

found = False
for match in response['FaceMatches']:
    print (match['Face']['FaceId'],match['Face']['Confidence'])
        
    face = dynamodb.get_item(
        TableName='facerecognition',  
        Key={'RekognitionId': {'S': match['Face']['FaceId']}}
        )
    
    if 'Item' in face:
        full_name = face['Item']['FullName']['S']
        print ("Found Person: ",full_name) 
        found = True
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
          # Append the full name and current time to the Attendance DataFrame
        Attendance = Attendance._append({'FullName': full_name, 'CurrentTime': current_time}, ignore_index=True)

        

if not found:
    print("Stranger")
    

# Save the updated DataFrame to the CSV file in append mode
Attendance.to_csv("attendance.csv", mode='a', index=False, header=False)
