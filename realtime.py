import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import glob
import sqlite3
from sqlite3 import Error
from datetime import datetime,timedelta
import time
#Veri Tabanı bağlantısı
conn =   sqlite3.connect(r"db.sqlite3")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

#Yüzleri öğretme 
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    #yüzlerin bulunduğu klasörü alıyor
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
       #klösördeki öğrenci yüzleri için yüz belirleme işlemi gerçekleştiriliyor.
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

           #resimde birden fazla yüz olup olmadığı kontrol ediliyor.
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # KNN sınıflandırıcısında ağırlık olarak kaç komşu kullanılacak
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # KNN sınıflandırıcı ile öğretme işlemi yapılıyor.
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

     #DOSYA YAZDIRIYOR
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf


#Tanıma yani tahmin işlemini gerçekleştiriyor.
def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Öğrenilmiş model yüklemesi yapılıyor.
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
            

   #resimdeki yüzlerin lokasyonları belirleniyor.
    X_face_locations = face_recognition.face_locations(X_frame) 

    if len(X_face_locations) == 0:
        return []

     #bilinen yüzlerle karşılaştırma 
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # KNN ile en iyi eşleşmeyi belirliyor.
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    #Belirlenen yüzleri tanınan ve tanınmayan olarak döndürüyor.
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def student_count(knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    closest_distances, indices_count = knn_clf.kneighbors(n_neighbors=1)
    return indices_count


def show_prediction_labels_on_image(frame, predictions , course_name, course_id, nowـdate, nowـtime):

    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        name = name.encode("UTF-8")
        student_no = str(name, "utf-8")
        
        if student_no == "unknown": 
           
           draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
           text_width, text_height = draw.textsize(name)
           draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
           display_text =  "Sorry We Cant recognize you please contact ogr for updating your new Photo "
           draw.text((left + 6, bottom - text_height - 5), display_text, fill=(255, 255, 255, 255))
        else:
            cur.execute("SELECT status FROM staj_attendance WHERE student_id_id=?  AND course_id_id=?", (student_no,course_id,))
            rows = cur.fetchall()
            if len(rows)==0:
                
                 draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
                 text_width, text_height = draw.textsize(name)
                 draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
                 display_text =  "ID:" + student_no + "Course:" + course_name + "iyi Dersler"
                 draw.text((left + 6, bottom - text_height - 15), display_text, fill=(0, 0, 0))
                 #veri tabanına yazılıyor.
                 sqlite_insert_query = """INSERT INTO staj_attendance (date, time, status, course_id_id, student_id_id) VALUES  (?, ?, ?, ?, ?);"""
                 data_tuple = (nowـdate, nowـtime, 1, course_id, student_no)
                 cur.execute(sqlite_insert_query, data_tuple)
                 conn.commit()
                 
                 print("Python Variables inserted successfully")
                  
            else: 
                 
                 draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                 text_width, text_height = draw.textsize(name)
                 draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
                 display_text =  "ID :" + student_no + "Course:" + course_name + "its done"
                 draw.text((left + 6, bottom - text_height - 10), display_text, fill=(0, 0, 0))
            
    del draw
    opencvimage = np.array(pil_image)
    return opencvimage

 
if __name__ == "__main__":
    
    while '1' == '1':
            now = datetime.now()
            last_student_count  = len(student_count(model_path="students.clf"))
            folder_num = glob.glob("students/*")
            len(folder_num)
            print("Number of subfolders in students are:"+str(len(folder_num)))
            if str(last_student_count) < str(folder_num):
                if  now.strftime("%H:%M:%S") == str('13:35:00') :
                        # if sub folder count in students folder ==  students.clf count  + "unknown"
                        print("Training KNN classifier...")
                        classifier = train("students", model_save_path="students.clf", n_neighbors=2)
                        print("Training complete!")
            cur = conn.cursor()
            now = datetime.now()
            nowـdate = now.strftime("%Y-%m-%d")
            nowـtime = now.strftime("%H:%M:%S")
            print(nowـdate)
            print(nowـtime)
    
            cur.execute("SELECT * FROM staj_course WHERE date=?  AND time=?", (nowـdate,nowـtime,))
            rows = cur.fetchall()
           
            if len(rows)==0:
                print('no class in this time to take attendance')
            else: 
                for row in rows:
                    course_id = row[0]
                    course_name = row[1]
                    course_date = row[2]
                    course_time = row[3]
                   
                    # process one frame in every 30 frames for speed
                    process_this_frame = 29
                    print('Setting cameras up...')

                    url = 'http://192.168.1.20:8080/video'
                    cap = cv2.VideoCapture(0)
                    
                    
                    while 1 > 0:
                        ret, frame = cap.read()
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,'ERCIYES  UNIVERSITESI', (50, 50), font, 1,(0, 255, 255),2,cv2.LINE_4)
                        
                        if ret:
                            
                            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            process_this_frame = process_this_frame + 1
                            if process_this_frame % 30 == 0:
                                predictions = predict(img, model_path="students.clf")
                            frame = show_prediction_labels_on_image(frame, predictions , course_name, course_id, nowـdate, nowـtime)
                            cv2.imshow('camera', frame)
                            if ord('q') == cv2.waitKey(10):
                                cap.release()
                                cv2.destroyAllWindows()
                                exit(0)
            time.sleep(1)


            # gelemen ogrenciler  30 de icerinndeki gelmeyenn ogrenciler attanced ekle ve satus 0 olarak eklenmeli