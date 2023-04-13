
    

from flask import Flask,jsonify,request
import cv2
import json
from datetime import date
from PIL import Image
import face_recognition
import numpy as np
# from flask_cors import CORS, cross_origin
import base64
# from firebase_admin import credentials, firestore, initialize_app
app=Flask(__name__)

# camera = cv2.VideoCapture(0)



# Initialize some variables


def gen_frames(frame,known_face_encodings,known_enrolls):    
    detected=False;
    matchAmount=1;
    enroll="Unknown";    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
 
    # Only process every other frame of video to save time

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # print(face_locations);
    if len(face_locations)==1 and len(known_face_encodings)>0:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
        # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index]<0.525 and best_match_index<len(known_enrolls):
                detected=True
                # print(best_match_index)
                enroll=known_enrolls[best_match_index]
                matchAmount=face_distances[best_match_index];
            # name = known_face_names[best_match_index]
    return ({
        'enroll':enroll,
        'detected':detected,
        'Number_Of_Faces':len(face_locations),
        'match_amount':matchAmount,
    })   
    

@app.route('/',methods=['GET'])
def home():
    return "welcome to attendance api - every child will study now"


@app.route('/test',methods=['GET','POST'])
def test():

    known_face_encodings=[]
    known_enrolls=[]
    data=base64.b64decode(request.json['image']);
    npimg = np.frombuffer(data, np.uint8);
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    # print(img)
    for encoding in request.json['encodings']:
        known_face_encodings.append(np.array(json.loads(encoding)))
    known_enrolls=request.json['enrolls']   

    data=gen_frames(img,known_face_encodings,known_enrolls);
    msg=""
    if data['Number_Of_Faces']>1:
        msg+="Multiple"
    elif data['Number_Of_Faces']==0:
        msg+="No Face"
    elif data['detected']==False:
        msg+="Mismatched"
    else:
        msg+="Matched"

    return jsonify({'enroll':data['enroll'],
                    'detected':data['detected'],
                    'number_of_faces':data['Number_Of_Faces'],
                    'message':msg,
                    'match_amount':data['match_amount']
    })


# route to register a person and get his facial encodings to store in firebase for future purposes
@app.route('/registration',methods=['GET','POST'])
def encode():
    data=base64.b64decode(request.json['image']);
    npimg = np.frombuffer(data, np.uint8);
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    face_locations = face_recognition.face_locations(img)
    face_encoding = face_recognition.face_encodings(img, face_locations)[0]
    face_encoding=json.dumps(face_encoding.tolist());
    return jsonify({
        'encoding':face_encoding,
    })

if __name__=="__main__":
    app.run(debug=True);