# DEEP MIND CONTROL SUITE: SHOW A VIDEO
# https://github.com/deepmind/dm_control/issues/39


from dm_control import suite
import numpy as np
import cv2

def grabFrame(env):
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

# Load task:
env = suite.load(domain_name="cartpole", task_name="swingup")

# Setup video writer - mp4 at 30 fps
video_name = 'video.mp4'
frame = grabFrame(env)
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

# First pass - Step through an episode and capture each frame
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    frame = grabFrame(env)
    # Render env output to video
    video.write(grabFrame(env))

# End render to video file
video.release()

# Second pass - Playback
cap = cv2.VideoCapture(video_name)
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('Playback', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# Exit
cv2.destroyAllWindows()