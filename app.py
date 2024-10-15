# from fastapi import FastAPI, Header, HTTPException
# import uvicorn
# import cv2
# import requests
# from contextlib import asynccontextmanager

# app = FastAPI()

# # Global variable for the camera capture
# cap = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global cap
#     cap = cv2.VideoCapture(0)  # Open the default camera (change index to 1 if using external webcam)
#     yield
#     cap.release()
#     cv2.destroyAllWindows()

# app = FastAPI(lifespan=lifespan)

# def send_image_to_backend(image_path, token):
#     """
#     Function to send the image to the Express.js backend server with the token.
#     """
#     headers = {"Authorization": f"Bearer {token}"}
#     with open(image_path, "rb") as image_file:
#         response = requests.post("http://localhost:3000/scan", files={"file": image_file}, headers=headers)
#     return response

# def detect_and_send_contours_to_backend(token):
#     """
#     Detects the largest object in the video feed, saves the image, and sends it to the Express backend.
#     """
#     # Open video capture from webcam (index 0 means default camera, change index for external camera)
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Capture each frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not capture frame from camera.")
#             break  # Break if the frame is not captured correctly

#         # Convert frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply a threshold to get a binary image
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#         # Find contours in the frame
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # If no contours found, continue to the next frame
#         if not contours:
#             continue

#         # Find the largest contour based on area
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # Filter based on aspect ratio and area (optional)
#         aspect_ratio = w / float(h)
#         area = cv2.contourArea(largest_contour)

#         detected = False
#         if 1.0 < aspect_ratio < 2.2 and 50000 < area < 100000:
#             # Extract the region of interest (ROI) for the largest contour
#             detected_object = frame[y:y + h, x:x + w]

#             # Save the largest contour as an image file
#             image_path = "captured_object.jpg"
#             cv2.imwrite(image_path, detected_object)
#             print(f"Saved largest object as {image_path}")

#             # Send the image to the Express.js backend
#             response = send_image_to_backend(image_path, token)

#             if response.status_code == 200:
#                 print(f"Image successfully sent to backend. Response: {response.json()}")
#             else:
#                 print(f"Failed to send image to backend. Status code: {response.status_code}")

#             # Display the image with a rectangle drawn around the detected object (optional)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             detected = True

#         # Display the live video with contours
#         cv2.imshow("Largest Object", frame)

#         if detected:
#             break

#         # Press 'q' to exit the loop and close the window
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# @app.post("/capture")
# async def capture(authorization: str = Header(...)):
#     """
#     Capture the image from the webcam, detect the largest contour, and send it to the Express backend.
#     """
#     # Use the authorization header directly as the token without any "Bearer " prefix
#     token = authorization.strip()

#     if not token:
#         raise HTTPException(status_code=400, detail="Token is required")

#     detect_and_send_contours_to_backend(token)
#     return {"message": "Image captured and sent to backend"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5000)

from fastapi import FastAPI, Header, HTTPException
import uvicorn
import cv2
import requests
from contextlib import asynccontextmanager

app = FastAPI()

# Global variable for the camera capture
cap = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera (change index to 1 if using external webcam)
    yield
    cap.release()
    cv2.destroyAllWindows()

app = FastAPI(lifespan=lifespan)

def send_image_to_backend(image_path, token):
    """
    Function to send the image to the Express.js backend server with the token.
    """
    headers = {"Authorization": f"Bearer {token}"}
    with open(image_path, "rb") as image_file:
        response = requests.post("http://localhost:3000/scan", files={"file": image_file}, headers=headers)
    return response

def detect_and_send_contours_to_backend(token):
    """
    Detects the largest object in the video feed, saves the image, and sends it to the Express backend.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture frame from camera.")
            break  # Break if the frame is not captured correctly

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(largest_contour)

        detected = False
        if 1.0 < aspect_ratio < 2.2 and 50000 < area < 100000:
            detected_object = frame[y:y + h, x:x + w]
            image_path = "captured_object.jpg"
            cv2.imwrite(image_path, detected_object)
            print(f"Saved largest object as {image_path}")
            response = send_image_to_backend(image_path, token)

            if response.status_code == 200:
                print(f"Image successfully sent to backend. Response: {response.json()}")
            else:
                print(f"Failed to send image to backend. Status code: {response.status_code}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected = True

        cv2.imshow("Largest Object", frame)

        if detected:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.post("/capture")
async def capture(authorization: str = Header(...)):
    """
    Capture the image from the webcam, detect the largest contour, and send it to the Express backend.
    """
    token = authorization.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    print(f"Received token: {token}")  # Debug print
    detect_and_send_contours_to_backend(token)
    return {"message": "Image captured and sent to backend"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
