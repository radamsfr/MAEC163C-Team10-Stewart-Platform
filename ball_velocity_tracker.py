import cv2
from matplotlib.pyplot import hsv
import numpy as np
import time

def track_ping_pong_ball():
    cap = cv2.VideoCapture(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # HSV bounds to track orange ping pong ball
    lower_orange = np.array([5, 150, 100])   
    upper_orange = np.array([25, 255, 255])

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Variables to track position and time across frames
    prev_center = None
    prev_time = time.time()

    print("Tracking started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)


        # to track a colored ball instead, convert to HSV
        
        # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create the mask for orange
        # mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Clean up small artifacts/shadows
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        
        
        # Threshold image (binary mask)
        _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_center = None
        velocity_x = 0.0
        velocity_y = 0.0
        
        if len(contours) > 0:
            valid_ball_contours = []
            
            for contour in contours:
                # Filter out tiny noise blobs by area
                area = cv2.contourArea(contour)
                if area < 100:  # Adjust based on how far away the ball sits
                    continue
                
                # Check Aspect Ratio of the bounding box
                x_box, y_box, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # A perfect circle has an aspect ratio of 1.0. Allow a small tolerance (e.g., 0.8 to 1.25)
                if not (0.8 <= aspect_ratio <= 1.25):
                    continue
                
                # Check Circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
                # A circle is 1.0. A square is ~0.78. Background clutter is usually much lower.
                if circularity > 0.75:
                    valid_ball_contours.append((contour, area))
            
            # pick largest valid contour as the ball (in case of multiple reflections or noise)
            if len(valid_ball_contours) > 0:
                largest_contour = max(valid_ball_contours, key=lambda item: item[1])[0]
                
                # Proceed with calculation using 'largest_contour'
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 0:
                    current_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Draw targets
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, current_center, 5, (0, 0, 255), -1)
                    
                    # Calculate Velocity (Pixels per Second)
                    if prev_center is not None and dt > 0:
                        dx = current_center[0] - prev_center[0]
                        dy = current_center[1] - prev_center[1]
                        
                        velocity_x = dx / dt
                        velocity_y = dy / dt

                        # Display velocity vector on the screen
                        cv2.line(frame, current_center, (current_center[0] + int(dx*2), current_center[1] + int(dy*2)), (255, 0, 0), 2)


        # --- OUTPUT FOR STEWART PLATFORM CONTROL ---
        if current_center is not None:
            # Draw targets
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, current_center, 5, (0, 0, 255), -1)

            # ONLY calculate velocity if the ball has actually shifted positions
            # filter out cases of no movement (prevent 0 vel.)
            
            if prev_center is not None and current_center != prev_center:
                dt = current_time - prev_time
                
                if dt > 0:
                    dx = current_center[0] - prev_center[0]
                    dy = current_center[1] - prev_center[1]
                    velocity_x = dx / dt
                    velocity_y = dy / dt
                    
                    # Update tracking timestamps only on a genuine new frame
                    prev_time = current_time
                    prev_center = current_center
                
                print(f"Ball Position: {current_center}, Velocity: ({velocity_x:.2f} px/s, {velocity_y:.2f} px/s)")
            
            elif prev_center is None:
                # Initialize first frame variables
                prev_center = current_center
                prev_time = current_time
                
        # Update previous center for the next frame calculation
        prev_center = current_center
                        

        # Display video feeds
        cv2.imshow("Tracking Feed", frame)
        cv2.imshow("Threshold (Binary)", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_ping_pong_ball()